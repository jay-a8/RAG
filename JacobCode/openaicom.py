import openai
import sys
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
# def api_key(key):
#     openai.api_key = key


def convert_to_topic(question):
    # if not openai.api_key:
    #     raise ValueError("API key not set. Use set_api_key() to set it.")

    if type(question) != str:
        question = str(question)

    response =  openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": """You are a helpful but concise assistant that takes a sentence and extracts the topics or most imporant subjects. Only extract the main subjects and stick to just a couple words.
            for example:
            user: \"What can cause a car accident?\"
            system: \"Car accident\"
            user: \"What is a basilica used for?\"
            system: \"Basilica\"
            user: \"What temperature limit of a car engine is safe?\"
            system: \"Car engine\"
            Convert this question: """ + question}],
        stream=True,
    )
    
    returned_string = ""

    for chunk in response:
        # Check if 'choices' and 'delta' exist, and ensure 'content' is present
        if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            returned_string += chunk['choices'][0]['delta']['content']

    return returned_string

def ask_chat_gpt(question):
    # if not openai.api_key:
    #     raise ValueError("API key not set. Use set_api_key() to set it.")

    if type(question) != str:
        question = str(question)


    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": """You are a helpful but concise assistant that answers questions simply and provides no explanation. For example:
            user: \"What can cause a car accident?\"
            system: Poor maintenance
            or user: \"What is a basilica used for?\"
            system: Religion
            Answer this question: """ + question}],
        stream=True,
    )
    
    returned_string = ""

    for chunk in response:
        # Check if 'choices' and 'delta' exist, and ensure 'content' is present
        if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            returned_string += chunk['choices'][0]['delta']['content']

    return returned_string





def convert_path_to_sentence(path):
    # if not openai.api_key:
    #     raise ValueError("API key not set. Use set_api_key() to set it.")

    if type(path) != str:
        path = str(path)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": """You are a helpful but concise assistant that converts an array of objects connected by relationships. You will use the array given to you to draw a logical path of reasoning from the start to end. You will return an explanation that is concise and makes logical sense connecting each element in the array given to you. For example:
            user: \"('Seats', 'has', 'Car'), ('Car', 'has', 'Engine'), ('Engine', 'needs', 'Fuel'), ('Fuel', 'type', 'Gas')\"
            system: All car's have an engine, engines require fuel to run, and gas is a type of fuel.
            Convert this path: """ + path}],
        stream=True,
    )
    
    returned_string = ""

    for chunk in response:
        # Check if 'choices' and 'delta' exist, and ensure 'content' is present
        if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
            returned_string += chunk['choices'][0]['delta']['content']

    return returned_string



##print(convert_path_to_sentence(sys.argv[1]))

def convert_to_kg(textblock):
    if type(textblock) != str:
        textblock = str(textblock)


    llm = OllamaLLM(model="llama3.2")

    prompt = ChatPromptTemplate.from_template(
         template="""You are a helpful assistant that converts a block of text into a JSON file knowledge graph that contains the main topics and relationships from the text. Use 'entities' and 'relationships' as the JSON sections and make sure you output the data as just a JSON within the {{}}. 
                    for example:
                    user: \"In the bustling city of Venora, nestled between the technological hub of Mardale and the historic district of Eastgate, Ella discovered a mysterious device in an old electronics shop, marked as an antique from the era of steam engines but equipped with advanced quantum circuitry. As she delved into its secrets, she found it could decrypt encrypted communications from the neighboring city's underground movement, a group aiming to disrupt the monopoly of TechGiant Corp over the city’s tech market. Her alliance with the movement not only threatened her safety but also poised her as a key figure in a burgeoning tech revolution, drawing the attention of both the rebels and corporate spies.\"
                    system: 
        {{
        "entities": [
            {{
            "id": "Ella",
            "type": "Character",
            }},
            {{
            "id": "Venora",
            "type": "City",
            }},
            {{
            "id": "Mardale",
            "type": "City",
            }},
            {{
            "id": "Eastgate",
            "type": "District",
            }},
            {{
            "id": "Mysterious Device",
            "type": "Technology",
            }},
            {{
            "id": "Underground Movement",
            "type": "Group",
            }},
            {{
            "id": "TechGiant Corp",
            "type": "Corporation",
            }},
            {{
            "id": "Rebels",
            "type": "Group",
            }},
            {{
            "id": "Corporate Spies",
            "type": "Group",
            }}
        ],
        "relationships": [
            {{
            "source": "Ella",
            "target": "Mysterious Device",
            "type": "found"
            }},
            {{
            "source": "Mysterious Device",
            "target": "Venora",
            "type": "located_in"
            }},
            {{
            "source": "Venora",
            "target": "Mardale",
            "type": "neighbor"
            }},
            {{
            "source": "Venora",
            "target": "Eastgate",
            "type": "includes"
            }},
            {{
            "source": "Mysterious Device",
            "target": "Underground Movement",
            "type": "decrypts_communications_for"
            }},
            {{
            "source": "Underground Movement",
            "target": "TechGiant Corp",
            "type": "opposes"
            }},
            {{
            "source": "Ella",
            "target": "Underground Movement",
            "type": "allies_with"
            }},
            {{
            "source": "Ella",
            "target": "Rebels",
            "type": "attracts_attention_from"
            }},
            {{
            "source": "Ella",
            "target": "Corporate Spies",
            "type": "attracts_attention_from"
            }}
        ]
        }}

        Convert this text to a knowledge graph JSON:
        {textblock}
        Please output the JSON **without** any additional text, only the JSON within {{}}

        """
    )
    chain = prompt | llm
    response = chain.invoke({"textblock": textblock})

    # returned_string = ""
    # print(response)
    # for chunk in response:
    #     # Check if 'choices' and 'delta' exist, and ensure 'content' is present
    #     if 'choices' in chunk and 'delta' in chunk['choices'][0] and 'content' in chunk['choices'][0]['delta']:
    #         returned_string += chunk['choices'][0]['delta']['content']

    return response


#print(convert_to_kg("Lemons are good"))
#print(convert_to_kg("In the bustling city of Venora, nestled between the technological hub of Mardale and the historic district of Eastgate, Ella discovered a mysterious device in an old electronics shop, marked as an antique from the era of steam engines but equipped with advanced quantum circuitry. As she delved into its secrets, she found it could decrypt encrypted communications from the neighboring city's underground movement, a group aiming to disrupt the monopoly of TechGiant Corp over the city’s tech market. Her alliance with the movement not only threatened her safety but also poised her as a key figure in a burgeoning tech revolution, drawing the attention of both the rebels and corporate spies."))
