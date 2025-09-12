# Download ollama models to scratch rather than the home directory
#OLLAMA_SCRATCH=/scratch/project_2001659/mvsjober/ollama
#conda create -n spiral-env python=3.10 -y  # Only needed once
#source activate spiral-env
#conda install -c conda-forge pyarrow
#pip install pandas numpy autogen matplotlib networkx ag2[openai]
# pip uninstall requests
# pip install requests
#OLLAMA_SCRATCH=./models
#export OLLAMA_MODELS=${OLLAMA_SCRATCH}

# Add ollama installation dir to PATH
#export PATH=/usr/local/pace-apps/manual/packages/ollama/0.9.0/bin:$PATH

# Simple way to start ollama. All the server outputs will appear in
# the slurm log mixed with everything else.
#ollama serve &

# If you want to direct ollama server's outputs to a separate log file
# you can start it like this instead
#mkdir -p ${OLLAMA_SCRATCH}/logs
OLLAMA_HOST=127.0.0.1:11435 ollama serve
# Capture process id of ollama server
OLLAMA_PID=$!

# Wait to make sure Ollama has started properly
sleep 5

# After this you can use ollama normally in this session

# Example: use ollama commands
#ollama pull llama3.1:8b
#ollama list
python3 spiral.py --topic immigration --network_path network_plots/homophily_-0.5_edges.csv

# Example: Try REST API
# curl http://localhost:11434/api/generate -d '{
#   "model": "llama3.1:8b",
#   "prompt":"Why is the sky blue?"
# }'


# At the end of the job, stop the ollama server
kill $OLLAMA_PID

# COMMAND TO RUN IN TERMINAL
#OLLAMA_HOST=127.0.0.1:11435 ollama serve & sleep 5 && python3 spiral.py --topic immigration --network_path network_plots/homophily_-0.5_edges.csv

OLLAMA_HOST=127.0.0.1:11435 ollama serve & sleep 5 && python3 spiral.py --topic popsicles --output-path trajectories/tenRuns/outputVanillaBiasWillingnessAgainst.txt
&& python3 spiral.py --topic popsicles --network_path network_plots/homophily_0.5_edges.csv --output-path trajectories/tenRuns/output0.5Homophily.txt


OLLAMA_HOST=127.0.0.1:11438 ollama serve & sleep 5 && python3 spiral.py --topic popsicles --network_path network_plots/homophily_1.0_edges.csv --output-path trajectories/tenRuns/oneHomophilyAllUsersSupport.txt