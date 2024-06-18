from volcenginesdkarkruntime import Ark
import CheckSql

#Specify your own Ark here
client = Ark(ak="******", sk="*******")

#send the database info to LLM model
with open('info.txt', 'r') as file:                                                                                                                                  
    info_content = file.read()
#print(info_content)


def generateQuery(query)->int:
    answer = ""
    stream = client.chat.completions.create(
    #specify the model you use here
    model="******",
    messages=[
        {
            "role": "user",
            "content": f"read these {info_content} and generate a sql query for me so that I could know {query} "
            #template % text,
        },
    ],
    stream=True,
    temperature=1,
    top_p=0.7,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        answer += chunk.choices[0].delta.content
        
    return CheckSql.search(answer)



