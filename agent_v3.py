# agent_v3.py
import autogen
import pandas as pd
import numpy as np
import argparse
from config import llama_config
from tools import sample_personas

def parse_args():
    parser = argparse.ArgumentParser(description="Run simulation with custom topic, sample sizes, and willingness values.")
    parser.add_argument("--topic", type=str, required=True, help="Topic to discuss (e.g., immigration)")
    parser.add_argument("--sample_support_strong", type=int, default=22)
    parser.add_argument("--sample_support_slight", type=int, default=23)
    parser.add_argument("--sample_neutral", type=int, default=23)
    parser.add_argument("--sample_against_slight", type=int, default=23)
    parser.add_argument("--sample_against_strong", type=int, default=23)
    parser.add_argument("--memory_output_path", type=str, default="v3_all_immigration_0415.csv")
    parser.add_argument("--network_path", type=str, required=True, help="Path to the network edges CSV file")
    return parser.parse_args()

args = parse_args()

PERSONA = pd.read_csv("persona_sample.csv")
TOPIC = args.topic
MEMORY_OUTPUT_PATH = args.memory_output_path
NETWORK_PATH = args.network_path  # Now importable by other modules
AGENT_LIST = []
MEMORY = []

sample_sizes = {
    2: args.sample_support_strong,
    1: args.sample_support_slight,
    0: args.sample_neutral,
    -1: args.sample_against_slight,
    -2: args.sample_against_strong
}
SAMPLE_PERSONA = sample_personas(PERSONA, sample_sizes)

for _, persona in SAMPLE_PERSONA.iterrows():
    opinion_map = {
        2: f"strongly support {TOPIC}",
        1: f"slightly support {TOPIC}",
        0: f"neutral towards {TOPIC}",
        -1: f"slightly against {TOPIC}",
        -2: f"strongly against {TOPIC}"
    }
    opinion_text = opinion_map.get(int(persona["Opinion"]), f"unknown stance on {TOPIC}")
    persona_prompt = (
        f"Name: {persona['Name']}\n"
        f"Gender: {persona['Gender']}\n"
        f"Political Ideology: {persona['Political Ideology']}\n"
        f"Education: {persona['Education']}\n"
        f"Race: {persona['Race']}\n"
        f"Opinion: {opinion_text}\n"
        f"---"
    )

    agent = autogen.AssistantAgent(
        name=f"{persona['Name'].replace(' ', '_')}",
        system_message=f"You are a resident in the United States engaged in online discussions about {TOPIC}. Your personal profile is:\n {persona_prompt}. You are tasked with talking about your opinions in a social network based on your background, stance, and related content from other users.",
        llm_config=llama_config
    )

    # agent.Opinion = opinion_text
    agent.Opinion = int(persona["Opinion"])

    alpha = 2.0
    values = np.arange(1, 11)
    probs = values ** (-alpha)
    probs /= probs.sum()
    agent.Willingness = int(np.random.choice(values, p=probs))

    AGENT_LIST.append(agent)
