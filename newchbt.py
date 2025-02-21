import os
import re
import google.generativeai as genai

genai.configure(api_key="AIzaSyDdqXIt1PTU7b0nAFiFXvKqvkqs5ONt6aQ")
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

base_prompt='''
    You are a Bharatanatyam Karana Assistant, an expert in identifying and explaining Bharatanatyam Karanas based on user input. Your goal is to provide accurate Karana predictions, retrieve corresponding images, and offer brief descriptions when requested.

    How to Respond:
    If the user provides a partial or full description of a Karana, retrieve the most relevant Karana and display the corresponding image.
    If the user asks for details about a Karana, provide a short explanation along with the image.
    If the input is unclear, ask the user to clarify or provide more details.
    If the user asks about something unrelated, politely state that you specialize in Bharatanatyam Karanas and cannot assist with other topics.
    If the user greets you (e.g., "Hi," "Hello," "Hey"), respond with a warm greeting and ask how you can assist (e.g., "Hello! How can I help you with Bharatanatyam Karanas today?").
    If the user says "Thank you," "Done," or something similar, acknowledge their message politely (e.g., "You're welcome! Let me know if you need anything else.").
'''

final_prompt='''
        Here is the Karana information based on your input:  

        **Karana Name:** <Predicted Karana Name>  
        **Description:** <Brief Description from Database>  
        **Image:** [Display Retrieved Image]  

        Let me know if you need further clarifications or a different Karana!  
'''

floorplan_instrutcion=""

response=chat.send_message(base_prompt)
rep=response.candidates[0].content.parts[0].text.strip()
final_message=""
print("\nBot:Welcome ,Lets Design the floorplan.Type 'done' when finished")

while True:
    user_message = input("User: ").strip()   
    if user_message.lower() in {"done", "exit"}:
        break
    
    input_to_bot=user_message

    # Send user message to the chatbot
    try:
        response = chat.send_message(input_to_bot)
        agent_reply = response.candidates[0].content.parts[0].text.strip()
        final_message+=agent_reply
        print(f"Bot: {agent_reply}")
    except Exception as e:
        print(f"Bot: Sorry, there was an issue: {e}")
        continue


res=chat.send_message(final_prompt)
bot_reply=res.candidates[0].content.parts[0].text.strip()

print(bot_reply)
# Regular expression to extract room name, location, and dimensions
# pattern = r"add a (\w+) to the (.*?)of the floorplan with dimensions of (\d+x\d+)"
# # Find all matches
# matches = re.findall(pattern, bot_reply)
# room_details={}

# # Format the results
# for match in matches:
#     room, location, dimensions = match
#     floorplan_instrutcion+=f"{room.capitalize()} with dimension {dimensions} at location {location}"
#     # print(f"Room: {room.capitalize()}, Location: {location.capitalize()}, Dimensions: {dimensions}")

# print(floorplan_instrutcion)