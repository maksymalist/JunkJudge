import pinecone

REGION = "us-east-1-aws"
API_KEY = "e32e440b-3009-46a8-9335-8492cdd51ee6"

pinecone.init(api_key=API_KEY,
              environment=REGION)

INDEX = pinecone.Index("junk-junction")


def add_entry(name, vector, metadata):
    upsert_response = INDEX.upsert(
        vectors=[
            (
                name,
                vector,
                metadata
            )    
        ],
        namespace="junk-junction",
    )
    print(upsert_response)
    
    
def query_embedding(vector, top_k=3, filter=None):
    vec = INDEX.query(
        vector=vector,
        top_k=top_k,
        include_values=True,
        filter=filter,
        include_metadata=True,
        namespace="junk-junction",
    )
    return vec