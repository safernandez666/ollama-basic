<h1 align="center"> 🦙 Basic Setup for Ollama & RAG </h1>


Welcome to the basic setup guide for Ollama & RAG! This guide will walk you through setting up your environment using Anaconda and loading dependencies from an `environment.yaml` file.


## Creating an Environment in Anaconda 

1. Install Anaconda: If you haven't already, you can download and install Anaconda from [its official website](https://www.anaconda.com/products/distribution).

2. Open Anaconda Navigator: Once installed, open Anaconda Navigator.

3. Create a New Environment:
   - Click on the "Environment" button on the left-hand menu.
   - Click on the "Create" button at the bottom left.
   - Give your environment a name (e.g., "ollama").
   - Select the Python version (in this case, Python 3.9).
   - Click "Create" to create the environment.

## Installing Dependencies from environment.yaml

1. Download the `environment.yaml` File: Make sure you have the `environment.yaml` file on your system.

2. Open Anaconda Prompt: Open Anaconda Prompt from the start menu.

3. Navigate to the Directory Containing `environment.yaml `: Use the `cd` command to navigate to the directory containing the `environment.yaml` file.

4. Install Dependencies **: Run the following command to install the dependencies specified in the `environment.yaml` file into the newly created environment:

## Install Ollama LLM

1. Download & Install from [here](https://ollama.com/download)

2. Download the mistral model

``` ollama pull mistral ```

## Explanation of the two programs

The first program needs to have a data folder, where the PDFs are saved in order to be added to the vector database. 

```python3 load.py```

<p align="center">
<img src="screenshots/load.png" width="800" >
</p>


The second one does the query with and without the vector database. To know if what we are doing is delivering an answer with proper context.

```python3 app.py```

<p align="center">
<img src="screenshots/app.png" width="800" >
</p>
