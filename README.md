## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
The multidocument retrieval agent was successfully implemented using LlamaIndex and OpenAI. Three space research papers were processed, and six tools were created, consisting of vector and summary tools for each paper. The agent successfully retrieved and synthesized information from the documents, generated summaries, and compared the research objectives of the three papers. Thus, the system demonstrated effective multidocument retrieval, summarization, comparison, and question answering across space research articles.
### DESIGN STEPS:

**STEP 1:** Download and load three space research papers related to exoplanet atmospheres, Mars exploration, and the James Webb Space Telescope (JWST) using their PDF URLs.

**STEP 2:** Process each research paper using LlamaIndex and create vector retrieval tools and summary tools for extracting relevant information from the documents.

**STEP 3:** Combine the retrieval and summary tools from all three papers and initialize an OpenAI-powered LlamaIndex function-calling agent.

**STEP 4:** Submit different queries to the agent, such as summarizing all papers and comparing their research objectives. The agent retrieves relevant information from the appropriate research documents.

**STEP 5:** Evaluate the generated responses based on relevance, accuracy, conciseness, and the ability to synthesize information across multiple documents.

### PROGRAM:
```

from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

# --------------------------------------------------
# SPACE RESEARCH PAPERS
# --------------------------------------------------

urls = [
    "https://arxiv.org/pdf/2408.09129",
    "https://link.springer.com/content/pdf/10.1007/s11214-022-00882-7.pdf",
    "https://ntrs.nasa.gov/api/citations/20250006005/downloads/36_Gardner_2023_PASP_135_068001.pdf",
]

papers = [
    "exoplanet_atmospheres.pdf",
    "mars_curiosity.pdf",
    "jwst_mission.pdf",
]


# --------------------------------------------------
# DOWNLOAD PAPERS
# --------------------------------------------------

import requests

for url, paper in zip(urls, papers):

    print(f"Downloading {paper}...")

    response = requests.get(url)

    with open(paper, "wb") as f:
        f.write(response.content)

    print(f"Downloaded: {paper}")


# --------------------------------------------------
# CREATE DOCUMENT TOOLS
# --------------------------------------------------

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}

for paper in papers:

    print(f"Getting tools for paper: {paper}")

    vector_tool, summary_tool = get_doc_tools(
        paper,
        Path(paper).stem
    )

    paper_to_tools_dict[paper] = [
        vector_tool,
        summary_tool
    ]


# --------------------------------------------------
# COMBINE ALL TOOLS
# --------------------------------------------------

initial_tools = [
    t
    for paper in papers
    for t in paper_to_tools_dict[paper]
]

print("Number of tools:", len(initial_tools))


# --------------------------------------------------
# OPENAI LLM
# --------------------------------------------------

from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY
)


# --------------------------------------------------
# CREATE AGENT
# --------------------------------------------------

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=True
)

agent = AgentRunner(agent_worker)


# --------------------------------------------------
# ASK QUESTIONS
# --------------------------------------------------

response = agent.query(
    """
    Give me a brief summary of all three space research papers.
    Mention the main topic, research objective, methodology,
    and important findings of each paper.
    """
)

print(str(response))
```
### OUTPUT:
```
Downloading exoplanet_atmospheres.pdf...
Downloaded: exoplanet_atmospheres.pdf
Downloading mars_curiosity.pdf...
Downloaded: mars_curiosity.pdf
Downloading jwst_mission.pdf...
Downloaded: jwst_mission.pdf
Getting tools for paper: exoplanet_atmospheres.pdf
Getting tools for paper: mars_curiosity.pdf
Getting tools for paper: jwst_mission.pdf
Number of tools: 6
Added user message to memory: 
    Give me a brief summary of all three space research papers.
    Mention the main topic, research objective, methodology,
    and important findings of each paper.
    
=== Calling Function ===
Calling function: summary_tool_exoplanet_atmospheres with args: {"input": "summary"}
=== Function Output ===
The study focuses on improving the characterization of exoplanet atmospheres by analyzing thermal emission spectra using physically motivated models for factors like water abundance profiles and cloud coverage. By comparing different retrieval models, the research aims to enhance the accuracy of estimating atmospheric properties such as radius, pressure-temperature structure, and gas abundances. The study emphasizes the importance of diverse models for temperate exoplanet atmospheres to understand how retrieval assumptions influence the interpretation of exoplanet spectra, aiding in the search for habitable worlds and potential signs of life with future space-based instruments.
=== Calling Function ===
Calling function: summary_tool_mars_curiosity with args: {"input": "summary"}
=== Function Output ===
The Mars Science Laboratory mission, with the Curiosity rover, has been exploring Gale crater on Mars since 2012. The mission aims to assess Mars' potential to support life by studying sediment deposits from ancient lakes that existed around 3.6 billion years ago. The rover has found evidence of past habitable environments, including the presence of organic molecules and diverse mineralogical compositions. The mission has also provided insights into Mars' ancient climate, atmospheric composition, and radiation levels. Curiosity's scientific instruments have enabled the collection of extensive data, leading to numerous publications and advancements in understanding Mars' geology, geochemistry, and potential for habitability. The mission's success is attributed to careful planning, operational flexibility, and a dedicated team culture.
=== Calling Function ===
Calling function: summary_tool_jwst_mission with args: {"input": "summary"}
=== Function Output ===
The James Webb Space Telescope (JWST) mission involves a diverse group of professionals from various institutions collaborating to ensure its success. Launched in December 2021, JWST is a 6.5m diameter cold space telescope orbiting the Sun-Earth L2 point. Equipped with four main science instruments - NIRCam, NIRSpec, MIRI, and NIRISS - JWST enables observations across different wavelengths, exploring cosmic history from the early universe to potentially habitable planets. Despite facing challenges like budget constraints and technical setbacks, the mission successfully completed its commissioning phase, with all instruments approved for scientific use by July 2022. JWST's observations have led to discoveries in areas such as galaxy evolution, early star formation, and exoplanet atmospheres, contributing significantly to current astronomical studies.
=== LLM Response ===
Here are brief summaries of the three space research papers:

1. **Exoplanet Atmospheres Research Paper:**
   - **Main Topic:** Improving the characterization of exoplanet atmospheres.
   - **Research Objective:** Analyzing thermal emission spectra using physically motivated models to estimate atmospheric properties accurately.
   - **Methodology:** Comparing different retrieval models to understand how retrieval assumptions influence the interpretation of exoplanet spectra.
   - **Important Findings:** Emphasizes the importance of diverse models for temperate exoplanet atmospheres to aid in the search for habitable worlds and potential signs of life with future space-based instruments.

2. **Mars Curiosity Research Paper:**
   - **Main Topic:** Exploration of Gale crater on Mars by the Curiosity rover.
   - **Research Objective:** Assessing Mars' potential to support life by studying sediment deposits from ancient lakes.
   - **Methodology:** Studying evidence of past habitable environments, including organic molecules and diverse mineralogical compositions.
   - **Important Findings:** Insights into Mars' ancient climate, atmospheric composition, radiation levels, and advancements in understanding Mars' geology, geochemistry, and potential for habitability.

3. **JWST Mission Research Paper:**
   - **Main Topic:** James Webb Space Telescope (JWST) mission and its diverse collaborations.
   - **Research Objective:** Observing cosmic history from the early universe to potentially habitable planets.
   - **Methodology:** Equipped with four main science instruments for observations across different wavelengths.
   - **Important Findings:** Discoveries in areas such as galaxy evolution, early star formation, and exoplanet atmospheres, contributing significantly to current astronomical studies.
assistant: Here are brief summaries of the three space research papers:

1. **Exoplanet Atmospheres Research Paper:**
   - **Main Topic:** Improving the characterization of exoplanet atmospheres.
   - **Research Objective:** Analyzing thermal emission spectra using physically motivated models to estimate atmospheric properties accurately.
   - **Methodology:** Comparing different retrieval models to understand how retrieval assumptions influence the interpretation of exoplanet spectra.
   - **Important Findings:** Emphasizes the importance of diverse models for temperate exoplanet atmospheres to aid in the search for habitable worlds and potential signs of life with future space-based instruments.

2. **Mars Curiosity Research Paper:**
   - **Main Topic:** Exploration of Gale crater on Mars by the Curiosity rover.
   - **Research Objective:** Assessing Mars' potential to support life by studying sediment deposits from ancient lakes.
   - **Methodology:** Studying evidence of past habitable environments, including organic molecules and diverse mineralogical compositions.
   - **Important Findings:** Insights into Mars' ancient climate, atmospheric composition, radiation levels, and advancements in understanding Mars' geology, geochemistry, and potential for habitability.

3. **JWST Mission Research Paper:**
   - **Main Topic:** James Webb Space Telescope (JWST) mission and its diverse collaborations.
   - **Research Objective:** Observing cosmic history from the early universe to potentially habitable planets.
   - **Methodology:** Equipped with four main science instruments for observations across different wavelengths.
   - **Important Findings:** Discoveries in areas such as galaxy evolution, early star formation, and exoplanet atmospheres, contributing significantly to current astronomical studies.
```

```
response = agent.query(
    "Compare the research objectives of all three space papers."
)

print(str(response))
```

```
Added user message to memory: Compare the research objectives of all three space papers.
=== Calling Function ===
Calling function: vector_tool_exoplanet_atmospheres with args: {"query": "research objectives"}
=== Function Output ===
The research objectives include demonstrating how adding simple physical constraints to the retrieval forward model can reduce biases in retrieved posteriors for an Earth-like spectrum, investigating the impact of assumptions made in the atmospheric models on the findings, exploring the sensitivity of the results to variations in trace-gas abundances, justifying the assumptions regarding cloud models and water abundance profiles, and utilizing a flexible atmospheric P-T profile model to fit any structure without imposing physical constraints.
=== Calling Function ===
Calling function: vector_tool_mars_curiosity with args: {"query": "research objectives"}
=== Function Output ===
The research objectives of the mission involve investigating specific sites to capture high-value observations, adapting to scientific discoveries and unforeseen issues, prioritizing mission science over distance milestones, and fostering an integrated team culture to maximize science return. The mission aims to balance time and depth of study, adjust timelines based on scientific findings, and focus on reaching key areas by fixed dates as the mission progresses. Additionally, the project emphasizes communication and collaboration between scientists and engineers to optimize instrument design and operations for higher scientific returns.
=== Calling Function ===
Calling function: vector_tool_jwst_mission with args: {"query": "research objectives"}
=== Function Output ===
The research objectives of the James Webb Space Telescope (JWST) include conducting medium-resolution spectroscopy, low-resolution slitless spectroscopy, high-contrast imaging, and imaging in broad-band filters and IFU spectroscopy across various wavelengths. The telescope aims to study a wide range of astronomical phenomena such as exoplanet atmospheres, Lyman-alpha emitting galaxies, galaxy evolution, and dusty sources in stellar environments.
=== LLM Response ===
The research objectives of the three space papers are as follows:

1. Exoplanet Atmospheres Paper:
   - Demonstrating how adding simple physical constraints to the retrieval forward model can reduce biases in retrieved posteriors for an Earth-like spectrum.
   - Investigating the impact of assumptions made in the atmospheric models on the findings.
   - Exploring the sensitivity of the results to variations in trace-gas abundances.
   - Justifying the assumptions regarding cloud models and water abundance profiles.
   - Utilizing a flexible atmospheric P-T profile model to fit any structure without imposing physical constraints.

2. Mars Curiosity Paper:
   - Investigating specific sites to capture high-value observations.
   - Adapting to scientific discoveries and unforeseen issues.
   - Prioritizing mission science over distance milestones.
   - Fostering an integrated team culture to maximize science return.
   - Balancing time and depth of study, adjusting timelines based on scientific findings, and focusing on reaching key areas by fixed dates as the mission progresses.
   - Emphasizing communication and collaboration between scientists and engineers to optimize instrument design and operations for higher scientific returns.

3. JWST Mission Paper:
   - Conducting medium-resolution spectroscopy, low-resolution slitless spectroscopy, high-contrast imaging, and imaging in broad-band filters and IFU spectroscopy across various wavelengths.
   - Studying a wide range of astronomical phenomena such as exoplanet atmospheres, Lyman-alpha emitting galaxies, galaxy evolution, and dusty sources in stellar environments.
assistant: The research objectives of the three space papers are as follows:

1. Exoplanet Atmospheres Paper:
   - Demonstrating how adding simple physical constraints to the retrieval forward model can reduce biases in retrieved posteriors for an Earth-like spectrum.
   - Investigating the impact of assumptions made in the atmospheric models on the findings.
   - Exploring the sensitivity of the results to variations in trace-gas abundances.
   - Justifying the assumptions regarding cloud models and water abundance profiles.
   - Utilizing a flexible atmospheric P-T profile model to fit any structure without imposing physical constraints.

2. Mars Curiosity Paper:
   - Investigating specific sites to capture high-value observations.
   - Adapting to scientific discoveries and unforeseen issues.
   - Prioritizing mission science over distance milestones.
   - Fostering an integrated team culture to maximize science return.
   - Balancing time and depth of study, adjusting timelines based on scientific findings, and focusing on reaching key areas by fixed dates as the mission progresses.
   - Emphasizing communication and collaboration between scientists and engineers to optimize instrument design and operations for higher scientific returns.

3. JWST Mission Paper:
   - Conducting medium-resolution spectroscopy, low-resolution slitless spectroscopy, high-contrast imaging, and imaging in broad-band filters and IFU spectroscopy across various wavelengths.
   - Studying a wide range of astronomical phenomena such as exoplanet atmospheres, Lyman-alpha emitting galaxies, galaxy evolution, and dusty sources in stellar environments.
```
### RESULT:
Traditional document search systems often process one document at a time, making it difficult to retrieve and combine relevant information from multiple research papers. Therefore, there is a need to develop a multidocument retrieval agent that can efficiently search multiple space research articles, identify relevant information, and synthesize it into concise and accurate responses. The system should support queries such as summarization, comparison, and cross-document question answering using LlamaIndex and OpenAI.
