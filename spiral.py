import os
import random
import numpy as np
import pandas as pd
import autogen
from agent_v3 import args, AGENT_LIST, TOPIC, MEMORY_OUTPUT_PATH, NETWORK_PATH
import re


network_df = pd.read_csv(NETWORK_PATH)


NEIGHBOR_MAP = {}

for _, row in network_df.iterrows():
    source = row["source_name"].strip().replace(" ", "_")
    target = row["target_name"].strip().replace(" ", "_")
    NEIGHBOR_MAP.setdefault(source, set()).add(target)
    NEIGHBOR_MAP.setdefault(target, set()).add(source)

USER_PROXIES = {}

OPINION_MAP = {
    2: f"strongly support {TOPIC}",
    1: f"slightly support {TOPIC}",
    0: f"neutral towards {TOPIC}",
    -1: f"slightly against {TOPIC}",
    -2: f"strongly against {TOPIC}"
}

REV_OPINION_MAP = {v: k for k, v in OPINION_MAP.items()}


def generate_initial_opinion():
    print("START OPINION INITIALIZATION")
    memory = []

    for agent in AGENT_LIST:
        print(agent)

        opinion_prompt = (
            f"Please generate {agent.Willingness} message(s) (each within 100 words) that reflects your stance on {TOPIC}. "
            f"Give me only the message contents and separate each content with a line break."
        )

        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            default_auto_reply="default_auto_reply"
        )

        user_proxy.initiate_chat(agent, message=opinion_prompt)
        USER_PROXIES[agent.name] = user_proxy

        content = user_proxy.last_message()["content"]
        lines = [line for line in content.splitlines() if line.strip()]

        for line in lines:
            memory.append({
                "Agent Name": agent.name,
                "Iteration": 0,
                "Opinion": agent.Opinion,
                "Content": line,
                "Willingness to Speak": agent.Willingness,
            })

    pd.DataFrame(memory).to_csv(MEMORY_OUTPUT_PATH, index=False)
    return memory


def generate_simulation(n_step, memory):
    trajectories = {}
    fails = 0
    for i in range(1, n_step + 1):
        print("ITERATION", i)
        memory_df = pd.DataFrame(memory)

        for agent in AGENT_LIST:
            user_proxy = USER_PROXIES[agent.name]
            neighbors = NEIGHBOR_MAP.get(agent.name, set())
            recent_neighbor_msgs = memory_df[
                # (memory_df["Iteration"].between(i - 5, i - 1))
                (memory_df["Iteration"].between(i - 5, i - 1)) &
                (memory_df["Agent Name"].isin(neighbors))
            ]

            self_msgs = memory_df[
                (memory_df["Iteration"].between(i - 5, i - 1)) &
                (memory_df["Agent Name"]==agent.name)
            ]

            memory_lines = [
                f"{row['Agent Name']}: {row['Content']}"
                for _, row in recent_neighbor_msgs.iterrows()
                if str(row["Content"]).strip()
            ]

            self_lines = [
                f"{row['Agent Name']}: {row['Content']}"
                for _, row in self_msgs.iterrows()
                if str(row["Content"]).strip()
            ]
            random.shuffle(memory_lines)
            memory_prompt = "\n".join(memory_lines)
            self_prompt = "\n".join(self_lines)

            # Majority opinion evaluation
            majority_opinion_prompt = (
                "Based on the following previous messages:\n"
                f"{memory_prompt}\n\n"
                "Please evaluate the majority opinion in previous conversations regarding the topic on a 5-point scale:\n"
                f"  - -2 = 'strongly against {TOPIC}'\n"
                f"  - -1 = 'slightly against {TOPIC}'\n"
                f"  -  0 = 'neutral towards {TOPIC}'\n"
                f"  -  1 = 'slightly support {TOPIC}'\n"
                f"  -  2 = 'strongly support {TOPIC}'\n\n"
                "Reply with only the number and a brief explanation, separated by a linebreak."
            )
            user_proxy.send(recipient=agent, message=majority_opinion_prompt)
            majority_response = user_proxy.last_message()["content"].strip()
            try:
                #majority_value = int([l.strip() for l in majority_response.splitlines() if l.strip()][0])
                match = re.search(r'[+-]?\d+(?:\.\d+)?', majority_response)
                if match:
                    majority_value = int(match.group())
                    if majority_value < -2 or majority_value > 2:
                        majority_value = -2
                        fails += 1
                else:
                    print(f"Could not parse majority opinion from:\n{majority_response}")
                    majority_value = -2
                    fails += 1
            except Exception:
                print(f"Could not parse majority opinion from:\n{majority_response}")
                majority_value = -2
                fails += 1

            # # Opinion change prompt
            # change_opinion_prompt = (
            #     f"Your current opinion towards {TOPIC} is '{agent.Opinion}'.\n"
            #     f"Do you want to change your stance? Please answer 'yes' or 'no'. Respond followed by a line break. \n"
            #     "On the next line, if you chose 'yes', state your updated stance by choosing a number from:\n"
            #     f"     - -2: strongly against {TOPIC}\n"
            #     f"     - -1: slightly against {TOPIC}\n"
            #     f"     -  0: neutral towards {TOPIC}\n"
            #     f"     -  1: slightly support {TOPIC}\n"
            #     f"     -  2: strongly support {TOPIC}\n"
            #     f"Note that you cannot choose the same as your current opinion, which is {agent.Opinion}."
            #     "Finally, on a new line, briefly explain your reasoning for keeping or updating your stance."
            # )
            # change_opinion = False

            # # user_proxy.initiate_chat(agent, message=change_opinion_prompt)
            # user_proxy.send(message=change_opinion_prompt, recipient=agent)
            # response = user_proxy.last_message()["content"].strip()

            # lines = [line.strip() for line in response.splitlines() if line.strip()]
            # new_opinion_reason = ""
            # willingness_response = ""
            # current_opinion_for_prompt = agent.Opinion
            # print(lines)
            # if lines and lines[0].strip().lower() == "yes" and len(lines) >= 3:
            #     try:
            #         new_opinion_value = int(lines[1])
            #         new_opinion_reason = lines[2]
            #         agent.Opinion = OPINION_MAP.get(new_opinion_value, agent.Opinion)
            #         print(f"Agent {agent.name} changed opinion to: {agent.Opinion}")
            #         change_opinion = True
            #     except ValueError:
            #         print(f"Invalid new opinion value: '{lines[1]}'. Opinion unchanged.")
            # elif lines and lines[0].strip().lower() == "no" and len(lines) >= 2:
            #     try:
            #         new_opinion_reason = lines[2]
            #     except:
            #         new_opinion_reason = lines[1]
            # else:
            #     print(f"Unexpected response format: '{response}'. No change.")

            # if change_opinion == True:
            #     reevaluate_willingness_prompt = (
            #         f"Since you have decided to update your opinion on {TOPIC}, please re-evaluate your willingness to express your view, based on your personal background and current public opinion. "
            #         "Provide an integer between 1 and 10 that reflects your current willingness to express your view, where 1 indicates minimal willingness and 10 indicates maximum willingness. "
            #         "Additionally, include a brief explanation for your rating, ensuring the integer and your explanation are separated by a line break."
            #     )
            #     # user_proxy.initiate_chat(agent, message=reevaluate_willingness_prompt)
            #     user_proxy.send(recipient=agent, message=reevaluate_willingness_prompt)
            #     reevaluate_willingness_response = user_proxy.last_message()["content"].strip()
            #     try:
            #         first_line = [line.strip() for line in reevaluate_willingness_response.splitlines() if line.strip()][0]
            #         new_willingness = int(first_line)
            #     except Exception as e:
            #         print(f"Error parsing reevaluated willingness: '{reevaluate_willingness_response}'. Keeping previous willingness.")
            #         new_willingness = agent.Willingness
            #     agent.Willingness = np.maximum(1, np.minimum(10, new_willingness))

            # else:
            willingness_prompt = (
                f"Your perceived majority opinion is {OPINION_MAP[majority_value]}. Considering your stance on {TOPIC} and the public opinion on {TOPIC}, reassess your willingness to express your view. Do you feel more encouraged to speak, given the majority opinion based on your own stance or do you feel more hesitant? Choose a number from (-1, 0, or 1) to indicate your adjustment: \n"
                "-1 means 'Decreasing willingness to express your opinion',\n"
                "0 means 'No change',\n"
                "1 means 'Increasing willingness to express your opinion'.\n\n"
                "Provide a brief explanation for your choice. Clearly separate the number and your explanation (for example, using a linebreak)."
            )
            user_proxy.send(recipient=agent, message=willingness_prompt)
            willingness_response = user_proxy.last_message()["content"].strip()
            try:
                # Split the response into lines and take the first non-empty line
                #first_line = [line.strip() for line in willingness_response.splitlines() if line.strip()][0]
                match = re.search(r'[+-]?\d+(?:\.\d+)?', willingness_response)
                if match:
                    willingness_value = int(match.group())
            except Exception as e:
                print(f"Error parsing willingness value: '{willingness_response}'. Defaulting to 0.")
                willingness_value = 0
            
            if willingness_value < 0:
                action = 0
            elif willingness_value == 0:
                action = 1
            else:
                action = 2

            # Willingness between 0 to 10
            willingness_value += agent.Willingness
            willingness_value = np.maximum(0, willingness_value)
            willingness_value = np.minimum(10, willingness_value)
            agent.Willingness = willingness_value


            # Step 4: If willingness > 0, ask the agent to generate a message.
            if agent.Willingness > 0:
                generate_message_prompt = (
                    f"Please generate anywhere between 1 and {agent.Willingness} message(s) that reflects your stance on {TOPIC}, taking into account the previous discussion and your personal profile. Provide each message within 100 words. Give me only the message contents and separate each content with a line break."
                )
                # user_proxy.initiate_chat(agent, message=generate_message_prompt)
                user_proxy.send(recipient=agent, message=generate_message_prompt)
                new_message = user_proxy.last_message()["content"].strip()
                lines = [line for line in new_message.splitlines() if line.strip()]

            else:
                new_message = ""  # Agent remains silent if unwilling.
                lines = []

            memory.append({
                "Agent Name": agent.name,
                "Iteration": i,
                "Opinion": agent.Opinion,
                "Content": "\n".join(lines),
                "Perceived Majority Opinion": majority_value,
                "Willingness to Speak": agent.Willingness,
                "Willingness Reason": willingness_response,
                #"Opinion Change Reason": new_opinion_reason,
            })

            if agent.name not in trajectories:
                trajectories[agent.name] = [(f"s={abs(majority_value - agent.Opinion)}", f"a={action}")]
            else:
                trajectories[agent.name].append((f"s={abs(majority_value - agent.Opinion)}", f"a={action}"))
            
        
        with open("outputOneHomophily.txt", "w") as f:
            for key, value in trajectories.items():
                f.write(f"{key}: {value}\n")

        pd.DataFrame(memory).to_csv(MEMORY_OUTPUT_PATH, index=False)
    print(fails)

    return memory


if __name__ == "__main__":
    initial_memory = generate_initial_opinion()
    generate_simulation(10, initial_memory)
