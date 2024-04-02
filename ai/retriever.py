import json
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone

try:
    with open(os.path.join(os.path.dirname(__file__), "secrets.json")) as f:
        secrets = json.loads(f.read())
except Exception:
    secrets = None
    print("!!! You should add secret.json file !!!")


def get_secret(key, secrets=secrets):
    return secrets[key]


def setup_pinecone():
    PINECONE_APIKEY = get_secret("PINECONE_APIKEY")
    pc = Pinecone(api_key=f"{PINECONE_APIKEY}", environment="gcp-starter")
    pc_index = pc.Index("cfn-doc")

    return pc_index


def setup_embedding():
    os.environ["OPENAI_API_KEY"] = get_secret("OPENAI_APIKEY")
    embedding = OpenAIEmbeddings()

    return embedding


def retrieve_docs(doc, title, vector_db, embedding, top_k):
    embedded_doc = embedding.embed_documents([doc])

    query_result = vector_db.query(
        vector=embedded_doc,
        top_k=top_k,
        filter={
            "title": {"$ne": title},
        },
        include_values=False,
        include_metadata=True
    )

    retrieved_docs = []

    for result in query_result.matches:
        title = result.metadata["title"]
        content = result.metadata["content"]
        score = result.score

        temp_result = {"title": title, "content": content, "score": score}
        retrieved_docs.append(temp_result)

    return retrieved_docs


if __name__ == "__main__":
    # pinecone
    pc_index = setup_pinecone()

    # openai
    embedding = setup_embedding()

    doc_example = """
# AWS::ACMPCA::인증서 Api패스스루

발급된 인증서에 배치할 X.509 인증서 정보를 포함합니다. `APIPassthrough` 또는 `APICSRPassthrough` 템플릿 변형을 선택해야 하며, 그렇지 않으면 이 매개변수가 무시됩니다. 

다른 소스에서 충돌하거나 중복된 인증서 정보를 제공하는 경우 AWS Private CA는 작업 순서 규칙을 적용하여 어떤 정보가 사용되는지 결정합니다.

## 통사론

AWS CloudFormation 템플릿에서 이 엔터티를 선언하려면 다음 구문을 사용합니다.

### JSON (영문)

```
{
  "Extensions" : Extensions,
  "Subject" : Subject
}
```

## 속성

`Extensions`
인증서에 대한 X.509 확장 정보를 지정합니다.
*필수 항목 *여부: 아니요
*유형*: 확장
*업데이트 필요 사항*: 바꿔 놓음

`Subject`
인증서 주체에 대한 정보를 포함합니다. 인증서의 주체 필드는 인증서의 공개 키를 소유하거나 제어하는 엔터티를 식별합니다. 엔터티는 사용자, 컴퓨터, 장치 또는 서비스일 수 있습니다. 제목에는 X.500 고유 이름(DN)이 포함되어야 합니다. DN은 RDN(Relative Distinguished Name)의 시퀀스입니다. RDN은 인증서에서 쉼표로 구분됩니다.   
*필수 항목 *여부: 아니요
*유형*: 주제
*업데이트 필요 사항*: 바꿔 놓음
    """

    doc_title = "aws-properties-acmpca-certificate-apipassthrough.md"

    result_docs = retrieve_docs(doc=doc_example, title=doc_title, vector_db=pc_index, embedding=embedding, top_k=3)

    for doc in result_docs:
        print(f"score: {doc['score']}\n\ntitle: {doc['title']}\n\ncontent: {doc['content']}\n\n")
