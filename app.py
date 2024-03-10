from flask import Flask, redirect, url_for, render_template, request
from functions import initialize_conversation, initialize_conv_reco, get_chat_model_completions, moderation_check,intent_confirmation_layer,dictionary_present,compare_laptops_with_user

import openai

openai.api_key = open("api_key.txt", "r").read().strip()

app = Flask(__name__)

conversation_bot = []
conversation = initialize_conversation()
introduction = get_chat_model_completions(conversation)
conversation_bot.append({'bot':introduction})
top_3_restaurants = None


@app.route("/")
def default_func():
    global conversation_bot, conversation, top_3_restaurants
    return render_template("index_invite.html", name_xyz = conversation_bot)

@app.route("/end_conv", methods = ['POST','GET'])
def end_conv():
    global conversation_bot, conversation, top_3_restaurants
    conversation_bot = []
    conversation = initialize_conversation()
    introduction = get_chat_model_completions(conversation)
    conversation_bot.append({'bot':introduction})
    top_3_restaurants = None
    return redirect(url_for('default_func'))

@app.route("/invite", methods = ['POST'])
def invite():
    global conversation_bot, conversation, top_3_restaurants, conversation_reco
    user_input = request.form["user_input_message"]
    prompt = 'Remember your system message and that you are an intelligent restaurant recommendation assistant. So, you only help with questions around restaurant.'
    moderation = moderation_check(user_input)
    if moderation == 'Flagged':
        return redirect(url_for('end_conv'))

    if top_3_restaurants is None:
        conversation.append({"role": "user", "content": user_input + prompt})
        conversation_bot.append({'user':user_input})

        response_assistant = get_chat_model_completions(conversation)

        moderation = moderation_check(response_assistant)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        confirmation = intent_confirmation_layer(response_assistant)

        moderation = moderation_check(confirmation)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        if "No" in confirmation:
            conversation.append({"role": "assistant", "content": response_assistant})
            conversation_bot.append({'bot':response_assistant})
        else:
            response = dictionary_present(response_assistant)

            moderation = moderation_check(response)
            if moderation == 'Flagged':
                return redirect(url_for('end_conv'))

            conversation_bot.append({'bot':"Thank you for providing all the information. Kindly wait, while I fetch the products: \n"})
            print("Response: ", response)
            top_3_restaurants = compare_laptops_with_user(response)           

            # validated_reco = recommendation_validation(top_3_restaurants)

            if len(top_3_restaurants) == 0:
                conversation_bot.append({'bot':"Sorry, we do not have restaurants nearby that match your requirements. Connecting you to a human expert. Please end this conversation."})

            conversation_reco = initialize_conv_reco(top_3_restaurants)
            recommendation = get_chat_model_completions(conversation_reco)

            moderation = moderation_check(recommendation)
            if moderation == 'Flagged':
                return redirect(url_for('end_conv'))

            conversation_reco.append({"role": "user", "content": "This is my user profile" + response})

            conversation_reco.append({"role": "assistant", "content": recommendation})
            conversation_bot.append({'bot':recommendation})

            print(recommendation + '\n')

    else:
        conversation_reco.append({"role": "user", "content": user_input})
        conversation_bot.append({'user':user_input})

        response_asst_reco = get_chat_model_completions(conversation_reco)

        moderation = moderation_check(response_asst_reco)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        conversation.append({"role": "assistant", "content": response_asst_reco})
        conversation_bot.append({'bot':response_asst_reco})
    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host= "0.0.0.0")