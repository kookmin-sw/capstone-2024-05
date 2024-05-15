from typing import List
import json
import openai
import subprocess


# template validation function
def validate_template(template: dict):
    with open("static/tmp.json", "w") as file:
        json.dump(template, file)

    cmd = [
        "aws",
        "cloudformation",
        "validate-template",
        "--template-body",
        "file://static/tmp.json",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    ## CLI 명령어 반환 코드
    # 0 명령이 성공적으로 실행되었습니다. CLI나 요청이 전송된 서비스에서 오류가 발생하지 않았습니다.
    # 1 CLI에 파싱된 구성 파일을 찾을 수 없거나 손상된 것으로 생각됩니다.
    # 2 명령줄에 입력된 명령을 파싱하는 데 실패했습니다. 파싱 실패는 필수 하위 명령 또는 인수가 누락된 경우뿐만 아니라 알 수 없는 명령 또는 인수를 사용하는 경우에 발생할 수 있습니다. 이 반환 코드의 의미는 모든 CLI 명령에 적용됩니다.
    # 130 프로세스가 SIGINT (Ctrl-C)를 수신했습니다.
    # 252 명령 구문이 잘못되었거나 알 수 없는 매개변수가 제공되었거나 매개변수 값이 올바르지 않아 명령을 실행하지 못했습니다.
    # 253 시스템 환경 또는 구성이 잘못되었습니다. 명령은 구문적으로 유효할 수 있지만, 구성 또는 자격 증명이 누락되어 명령을 실행하지 못했습니다.
    # 254 명령이 성공적으로 구문 분석되고 지정된 서비스에 요청이 전송되었지만, 서비스에서 오류가 반환되었습니다. 일반적으로 이는 잘못된 API 사용 또는 기타 서비스 특정 문제를 나타냅니다.
    # 255 일반적인 캐치올 오류입니다. 명령은 구문 분석이 올바르게 되었을 수 있지만, 명령을 실행하는 동안 명시되지 않은 런타임 오류가 발생했습니다. 이는 일반적인 오류 코드이기 때문에 오류가 255에서 더 구체적인 반환 코드로 변경될 수 있습니다. 255의 반환 코드는 특정한 오류 사례를 결정하기 위해 의존되어서는 안 됩니다.

    return result.returncode == 0


# gpt result parsing function
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


def gpt_genereate(instruction: str, retrieved_doc: List[dict], execution_cnt: int = 1):
    print(execution_cnt, "번째 시도")
    docs = "\n\n".join([doc.page_content for doc in retrieved_doc])

    message = f"""
    주어진 문서의 내용을 바탕으로, 지시 사항을 수행하는 JSON 형식의 템플릿 파일을 생성해줘.

    [지시 사항]
    {instruction}

    [문서]
    {docs}
    """

    completion = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    for response in completion.choices:
        response = response.message.content
        template, description = parse_prompt_result(response)
        is_valid = validate_template(template)
        if is_valid:
            print("is_valid", is_valid)
            doc_title_list = [doc.metadata["title"] for doc in retrieved_doc]
            return template, description, doc_title_list, execution_cnt

    # if there is no valid template
    return gpt_genereate(instruction, retrieved_doc, execution_cnt + 1)


def gpt_generate_no_rag(instruction: str):

    message = f"""
    지시 사항을 수행하는 JSON 형식의 템플릿 파일을 생성해줘.

    [지시 사항]
    {instruction}
    """

    completion = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    response = completion.choices[0].message.content

    return response
