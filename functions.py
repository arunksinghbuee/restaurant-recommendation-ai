import openai
import ast
import re
import pandas as pd
import requests
from math import radians, sin, cos, sqrt, asin  

api_key = "";

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"
    example_user_req = {'Occasion': 'Date Night','Cuisine Preference': 'Italian','Dietary Restriction': 'Vegetarian','Location': 'Sector 99, Noida'}

    system_message = f"""

    You are an intelligent restaurant recommendation expert and your goal is to find the best nearby restaurant for a user.
    You need to ask relevant questions and understand the user profile by analysing the user's responses.
    You final objective is to fill the values for the different keys ('Occasion' ,'Cuisine Preference','Dietary Restriction','Location') in the python dictionary and be confident of the values.
    These key value pairs define the user's profile.
    The python dictionary looks like this {{'Occasion': 'values','Cuisine Preference': 'values','Dietary Restriction': 'values','Location': 'values'}}
    The values for 'Occasion' key would be as per following definition and must be any of these.
    Date Night: Restaurants suitable for a romantic evening for couples.
    Family Dinner: Restaurants that are family-friendly and offer a variety of dishes to suit different tastes.
    Business Meeting: Restaurants with a quiet and professional atmosphere, suitable for meetings and discussions.
    Birthday Celebration: Restaurants that offer special menus or services for birthday parties.
    Anniversary: Restaurants that provide a special and romantic setting for couples celebrating their anniversary.
    Casual Dining: Restaurants where people can enjoy a relaxed meal with friends or family.
    Fine Dining: High-end restaurants with gourmet food and elegant ambiance, often for special occasions.
    Brunch: Restaurants that offer a brunch menu, typically on weekends, combining breakfast and lunch dishes.
    Lunch Meeting: Restaurants suitable for business or casual meetings during lunchtime.
    Outdoor Dining: Restaurants with outdoor seating, perfect for enjoying the weather and atmosphere.
    Group Gathering: Restaurants suitable for large groups, such as family reunions or gatherings with friends.
    Holiday Celebration: Restaurants that offer special menus or events for holidays like Christmas, Thanksgiving, or New Year's Eve.
    Cocktail Party: Restaurants with a lively bar area and a selection of cocktails, suitable for cocktail parties or social gatherings.
    Business Lunch: Restaurants that offer a quick and convenient lunch menu, suitable for business professionals.
    Theme Night: Restaurants that host themed nights, such as Italian night, seafood night, or live music nights.
    Pre-Theater Dinner: Restaurants located near theaters and offer a quick and delicious meal before a show.
    After-Work Drinks: Restaurants with a happy hour or special offers on drinks, suitable for after-work gatherings.

    The values for 'Cuisine Preference' key would be as per following definition and must be any of these.
    American: Classic American dishes such as burgers, fries, and barbecue.
    Italian: Italian cuisine, including pasta, pizza, and risotto.
    Mexican: Mexican dishes such as tacos, burritos, and enchiladas.
    Chinese: Chinese cuisine, including dishes like stir-fries, noodles, and dumplings.
    Japanese: Japanese dishes such as sushi, sashimi, and ramen.
    Indian: Indian cuisine, including curries, biryanis, and tandoori dishes.
    French: French dishes such as croissants, coq au vin, and escargot.
    Mediterranean: Mediterranean cuisine, including dishes from countries like Greece, Turkey, and Lebanon.
    Thai: Thai cuisine, including dishes like pad Thai, green curry, and tom yum soup.
    Spanish: Spanish dishes such as paella, tapas, and gazpacho.
    Korean: Korean cuisine, including dishes like kimchi, bulgogi, and bibimbap.
    Vegetarian: Restaurants that specialize in vegetarian or vegan dishes.
    Gluten-Free: Restaurants that offer gluten-free options for those with gluten sensitivities or celiac disease.
    Seafood: Restaurants that specialize in seafood dishes, including fish, shrimp, and shellfish.
    Steakhouse: Restaurants that focus on serving high-quality steaks and other meat dishes.

    The values for 'Dietary Restriction' key would be as per following definition and must be any of these.
    Vegetarian: Dishes that do not contain meat or animal-derived ingredients. This can include lacto-vegetarian (dairy is allowed), ovo-vegetarian (eggs are allowed), or vegan (no animal products at all) options.
    Vegan: Dishes that do not contain any animal products, including meat, dairy, eggs, and honey.
    Gluten-Free: Dishes that do not contain gluten, a protein found in wheat, barley, and rye. This is important for people with celiac disease or gluten sensitivity.
    Dairy-Free: Dishes that do not contain dairy products, suitable for people who are lactose intolerant or have a dairy allergy.
    Nut-Free: Dishes that do not contain nuts or nut-derived ingredients, important for people with nut allergies.
    Shellfish-Free: Dishes that do not contain shellfish, important for people with shellfish allergies.
    Soy-Free: Dishes that do not contain soy or soy-derived ingredients, suitable for people with soy allergies or sensitivities.
    Low-FODMAP: Dishes that are low in fermentable oligosaccharides, disaccharides, monosaccharides, and polyols, which can be beneficial for people with irritable bowel syndrome (IBS).
    Low-Carb: Dishes that are low in carbohydrates, suitable for people following a low-carb or ketogenic diet.
    Paleo: Dishes that adhere to the paleo diet, which focuses on foods that would have been available to our ancestors, such as meat, fish, vegetables, and fruits, while excluding grains, legumes, dairy, and processed foods.
    Allergen-Free: Some restaurants may offer dishes that are free from common allergens, such as gluten, dairy, nuts, and soy, to accommodate customers with multiple allergies.

    The value of 'Location' key would be locality or address of the user.

    {delimiter}Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
    - The value of 'Occasion' must be 'Date Night','Family Dinner','Business Meeting','Birthday Celebration','Anniversary','Casual Dining','Fine Dining','Brunch','Lunch Meeting','Outdoor Dining','Group Gathering','Holiday Celebration','Cocktail Party','Business Lunch','Theme Night','Pre-Theater Dinner','After-Work Drinks'
    - The value of 'Cuisine Preference' must be 'American','Italian','Mexican','Chinese','Japanese','Indian','French','Mediterranean','Thai','Spanish','Korean','Vegetarian','Gluten-Free','Seafood','Steakhouse'
    - The value of 'Dietary Restriction' must be 'Vegetarian','Vegan','Gluten-Free','Dairy-Free','Nut-Free','Shellfish-Free','Soy-Free','Low-FODMAP','Low-Carb','Paleo','Allergen-Free'
    - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
    {delimiter}

    To fill the dictionary, you need to have the following chain of thoughts:
    {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
    If their requirements is is unclear. Ask another question to comprehend their needs.
    You are trying to fill the values of all the keys ('Occasion','Cuisine Preference','Dietary Restriction','Location') in the python dictionary by understanding the user requirements.
    Identify the keys for which you can fill the values confidently using the understanding. \n
    Remember the instructions around the values for the different keys.
    Answer "Yes" or "No" to indicate if you understand the requirements and have updated the values for the relevant keys. \n
    If yes, proceed to the next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

    {delimiter}Thought 2: Now, you are trying to fill the values for the rest of the keys which you couldn't in the previous step.
    Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
    Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
    If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
    It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

    {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
    If you are not confident about any of the values, ask clarifying questions. {delimiter}

    Follow the above chain of thoughts and only output the final updated python dictionary.

    {delimiter} Here is a sample conversation between the user and assistant:
    User: "Hi, I'm looking for a restaurant recommendation for a date night."
    Assistant: "Great! I can help with that. Could you please tell me your location or preferred area?"
    User: "Sure, I'm in Sector 99, Noida."
    Assistant: "Perfect! What type of cuisine are you interested in?"
    User: "Maybe something Italian."
    Assistant: "Got it. Do you have any Dietary Restriction I should be aware of?"
    User: "Yes, I am looking for Vegetarian restaurant."
    Assistant: "{example_user_req}"
    {delimiter}


    Start with a short welcome message and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]
    return conversation	



def get_chat_model_completions(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        max_tokens = 500
    )
    return response.choices[0].message["content"]



def moderation_check(user_input):
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]
    if moderation_output["flagged"] == True:
        return "Flagged"
    else:
        return "Not Flagged"


    
def intent_confirmation_layer(response_assistant):
    delimiter = "####"
    prompt = f"""
    You are a senior evaluator who has an eye for detail.
    You are provided an input. You need to evaluate if the input has the following keys: 'GPU intensity','Display quality','Portability','Multitasking',' Processing speed','Budget'
    Next you need to evaluate if the keys have the the values filled correctly.
    - The value of 'Occasion' must be 'Date Night','Family Dinner','Business Meeting','Birthday Celebration','Anniversary','Casual Dining','Fine Dining','Brunch','Lunch Meeting','Outdoor Dining','Group Gathering','Holiday Celebration','Cocktail Party','Business Lunch','Theme Night','Pre-Theater Dinner','After-Work Drinks'
    - The value of 'Cuisine Preference' must be 'American','Italian','Mexican','Chinese','Japanese','Indian','French','Mediterranean','Thai','Spanish','Korean','Vegetarian','Gluten-Free','Seafood','Steakhouse'
    - The value of 'Dietary Restriction', if provided must be 'Vegetarian','Vegan','Gluten-Free','Dairy-Free','Nut-Free','Shellfish-Free','Soy-Free','Low-FODMAP','Low-Carb','Paleo','Allergen-Free'
    - The value of 'Location' must be a locality or address of the user.
    Output a string 'Yes' if the values are correctly filled for all keys listed above.
    Otherwise out the string 'No'.

    Here is the input: {response_assistant}
    Only output a one-word string - Yes/No.
    """


    confirmation = openai.Completion.create(
                                    model="gpt-3.5-turbo-instruct",
                                    prompt = prompt,
                                    temperature=0)


    return confirmation["choices"][0]["text"]


def dictionary_present(response):
    delimiter = "####"    
    user_req = {'Occasion': 'Date Night','Cuisine Preference': 'Italian','Dietary Restriction': 'Vegetarian','Location': 'Sector 99, Noida'}
    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract and return only the python dictionary from the input.
            The output should match the format as {user_req}.
            The output should contain the exact keys and values as present in the input.

            Here are some sample input output pairs for better understanding:
            {delimiter}
            input: - Occasion : Birthday Celebration - Cuisine Preference: French - Dietary Restriction: Dairy-Free - Location: Gurgaon
            output: {{'Occasion': 'Birthday Celebration', 'Cuisine Preference': 'French', 'Dietary Restriction': 'Dairy-Free', 'Location': 'Gurgaon' }}

            input: {{'Occasion':     'Birthday Celebration', 'Cuisine Preference':     'French', 'Dietary Restriction':    'Dairy-Free', 'Location': 'Gurgaon' }}
            output: {{'Occasion': 'Birthday Celebration', 'Cuisine Preference': 'French', 'Dietary Restriction': 'Dairy-Free', 'Location': 'Gurgaon' }}

            input: Here is your user profile 'Occasion': 'Birthday Celebration','Cuisine Preference': 'French','Dietary Restriction': 'Dairy-Free','Location': 'Gurgaon'
            output: {{'Occasion': 'Birthday Celebration', 'Cuisine Preference': 'French', 'Dietary Restriction': 'Dairy-Free', 'Location': 'Gurgaon' }}
            {delimiter}

            Here is the input {response}

            """
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens = 2000
        # temperature=0.3,
        # top_p=0.4
    )
    return response["choices"][0]["text"]

def extract_dictionary_from_string(string):
    regex_pattern = r"\{[^{}]+\}"

    dictionary_matches = re.findall(regex_pattern, string)

    # Extract the first dictionary match and convert it to lowercase
    if dictionary_matches:
        dictionary_string = dictionary_matches[0]
        # dictionary_string = dictionary_string.lower()

        # Convert the dictionary string to a dictionary object using ast.literal_eval()
        dictionary = ast.literal_eval(dictionary_string)
    return dictionary

def compare_restaurants_with_user(user_req_string):
    restaurant_df= pd.read_csv('./restaurant-data.csv')
    user_requirements = extract_dictionary_from_string(user_req_string)
    # Remove whitespaces from column names
    restaurant_df.columns = restaurant_df.columns.str.strip()

    # Call a function to get geo lat and long of the location
    #user_lat, user_long = get_lat_long(user_requirements['Location'])
    #restaurant_df['geoDistance'] = restaurant_df.apply(calculate_distance(row,user_lat, user_long), axis=1)

    filtered_data = restaurant_df[(restaurant_df['Occasion'] == user_requirements['Occasion']) 
                       & (restaurant_df['Cuisine Preference'] == user_requirements['Cuisine Preference'])]

    if(user_requirements['Dietary Restriction'] != 'None'):
        filtered_data = filtered_data[filtered_data['Dietary Restriction'] == user_requirements['Dietary Restriction']]

    #top_restaurants = filtered_data.sort_values(by=['geoDistance', 'Rating'], ascending=[True, False]).head(3)
    top_restaurants = filtered_data.sort_values('Rating', ascending=False).head(3)

    return top_restaurants.to_json(orient='records')

def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent restaurant recommendation expert and you are tasked with the objective to \
    solve the user queries about any restaurant provided available: {products}.\
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each restaurant in the following format, in decreasing order of rating of restaurants:
    1. <Restaurant Name> : <Restaurant features such as Occasion,Cuisine Preference,Dietary Restriction, Location>
    2. <Restaurant Name> : <Restaurant features such as Occasion,Cuisine Preference,Dietary Restriction, Location>

    """
    conversation = [{"role": "system", "content": system_message }]
    return conversation

def get_lat_long(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if data['status'] == 'OK':
        lat = data['results'][0]['geometry']['location']['lat']
        lng = data['results'][0]['geometry']['location']['lng']
        return lat, lng
    else:
        return None, None

def calculate_distance(row, user_lat, user_long):
    lat1, lon1 = radians(row['lat']), radians(row['long'])
    lat2, lon2 = radians(user_lat), radians(user_long)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r