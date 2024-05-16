import json
import re
import os
import sys
from datetime import datetime
from typing import Dict

from db.database import get_db
from db.models import HashTag, PromptAns, PromptAns_HashTag
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from retrieval.rag import retrieve_doc
from sqlalchemy import insert, select
from sqlalchemy.orm import Session
from utils.gpt import gpt_genereate
from utils.auth import get_current_user

from utils.gpt import validate_template

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import main as app_main

router = APIRouter()


# function to get values of dict
def recursive_value_collect(dict_obj):
    for value in dict_obj.values():
        if isinstance(value, dict):
            yield from recursive_value_collect(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from recursive_value_collect(item)
                else:
                    yield item
        else:
            yield value


def parse_prompt_result(result):
    template_file, description = [], []
    state = False

    for l in result.split("\n"):
        if "```" in l:
            state = not state
            continue
        if state:
            template_file.append(l + "\n")
        else:
            description.append(l + "\n")

    description = "".join(description)
    description = description.strip()

    template_file = "".join(template_file)
    template_file = json.loads(template_file)

    return template_file, description


# 해시태그 추가함수
def add_hashtag(template):
    hashtags = template.hashtags
    hashtags = list(map(lambda x: x.tag, hashtags))
    if hashtags:
        # dictionary로 변환
        template = {
            c.name: getattr(template, c.name) for c in template.__table__.columns
        }
        # 해시태그 필드 추가
        template["hashtag"] = hashtags
    return template


# Data Validation
class TemplateUploadBase(BaseModel):
    id: int


class PromptAnsBase(BaseModel):
    prompt: str
    template: Dict
    description: str
    user_id: int  # FK


class Template(BaseModel):
    template: Dict


class Prompt(BaseModel):
    prompt: str


########### GET ###########
@router.get("")
async def get_templates(
    db: Session = Depends(get_db), username: str = Depends(get_current_user)
):
    try:
        if not username:  # jwt token이 없을 때
            templates = db.query(PromptAns).filter(PromptAns.uploaded.isnot(None)).all()
            templates = [add_hashtag(template) for template in templates]
            return templates
        else:
            templates = db.query(PromptAns).filter(PromptAns.username == username).all()
            templates = [add_hashtag(template) for template in templates]
            return templates
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)


@router.get("/{id}")
async def get_single_templates(
    id: int, db: Session = Depends(get_db), username: str = Depends(get_current_user)
):
    template = db.query(PromptAns).filter(PromptAns.id == id).first()

    if template:
        uploaded = template.uploaded
        if uploaded:
            template = add_hashtag(template)
            return template
        else:
            # - jwt 유무 판단.
            if not username:
                # - jwt가 없으면 에러
                raise HTTPException(
                    status_code=401,
                    detail="Could not validate credentials",
                )
            else:
                # - 본인이 작성한 템플릿이 아니면 에러발생
                if username != template.username:
                    raise HTTPException(
                        status_code=401,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    # - 생성자가 본인이면 리턴
                    template = add_hashtag(template)
                    return template
    else:
        raise HTTPException(status_code=404, detail="Template not found, invalid id")


########### POST ###########
@router.post("")
async def create_template(
    prompt: Prompt,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    try:
        prompt = prompt.prompt

        # make retrieved_doc
        retrieved_doc = retrieve_doc(
            question=prompt,
            pc_index=app_main.pc_index,
            embedding=app_main.embedding,
            llm=app_main.llm,
        )

        # generate template
        template_file, description, document_title_list, execution_cnt = gpt_genereate(
            instruction=prompt, retrieved_doc=retrieved_doc
        )

        if template_file is None:
            raise HTTPException(
                status_code=500,
                detail="[template] 생성 실패3회 시도에서 유효한 템플릿이 나오지 않았습니다. 프롬프트를 수정해주세요",
            )

        # get user_id from JWT
        db_promptAns = PromptAns(
            prompt=prompt,
            template=template_file,
            description=description,
            documents=document_title_list,
            execution_cnt=execution_cnt,
            username=username if username else None,  # jwt token이 없을 때 None
        )
        db.add(db_promptAns)
        db.commit()
        db.refresh(db_promptAns)

        ##### keword extraction from template #####
        # value 수집
        values = list(recursive_value_collect(template_file))
        values = "".join(map(str, values))
        # 정규 표현식을 사용하여 단어 추출
        keywords = re.findall(r"\b[A-Za-z]+\b", values)

        # 필터링
        filtered_keywords = [
            k for k in keywords if k not in ["AWS", "Amazon"] and len(k) > 2
        ]

        if not filtered_keywords:
            # 사전 정의된 키워드 기반이라 키워드가 없는 경우가 존재함
            return db_promptAns
        else:
            tag_ids = []
            # find keyword in db. 이미 존재하는 키워드만 필터링
            for keyword in filtered_keywords:
                hashtag = (
                    db.query(HashTag).filter(HashTag.tag.ilike(f"%{keyword}%")).first()
                )
                if hashtag is not None:
                    tag_ids.append(hashtag.id)

            # 새로운 키워드 추가
            for tag_id in tag_ids:
                stmt = insert(PromptAns_HashTag).values(
                    prompt_ans_id=db_promptAns.id, hashtag_id=tag_id
                )
                db.execute(stmt)
            db.commit()
            ##### return 값에 Hashtag 추가하기
            tags = list(
                map(
                    lambda x: x[0],
                    db.query(HashTag.tag).filter(HashTag.id.in_(tag_ids)).all(),
                )
            )

            # db_promptAns를 dictionary로 변환
            db_promptAns_dict = {
                c.name: getattr(db_promptAns, c.name)
                for c in db_promptAns.__table__.columns
            }
            # 해시태그 필드 추가
            db_promptAns_dict["hashtag"] = tags

            # 리턴
            return db_promptAns_dict
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=e)


@router.post("/upload")
def upload_template(
    params: TemplateUploadBase,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    try:
        if not username:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        prompt_ans = db.query(PromptAns).filter(PromptAns.id == params.id).first()
        if not prompt_ans:
            raise HTTPException(status_code=404, detail="id not found")
        if username != prompt_ans.username:
            raise HTTPException(status_code=404, detail="user name dose not match")
        prompt_ans.uploaded = datetime.now()
        db.commit()
        return {"detail": "upload success"}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=e)


@router.post("/validate")
def validate(params: Template):
    try:
        is_valid = True if validate_template(params.template) else False
        message = (
            "유효한 template 입니다." if is_valid else "유효하지 않은 template 입니다."
        )
        return {
            "isValid": is_valid,
            "message": message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
