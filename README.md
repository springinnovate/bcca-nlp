Research Paper Review Assistant using NLP
=========================================

This pipeline is designed to automate the process of asking specific questions about a collection of research papers using natural language processing. The pipeline processes a directory of research papers and provides answers to the specified questions for each paper.

Features
--------

- **Directory-Based Input:** The user provides a directory containing research papers in PDF or text format.
- **Customizable Questions:** The user specifies a set of questions to be asked about each paper.
- **NLP Processing:** The pipeline uses NLP models to analyze the content of each paper and generate answers to the provided questions.
- **Output Options:** The results are saved in a structured format for further analysis or reporting.

Usage
-----

1. Prepare the Input Directory

   Place all the research papers you want to process in a directory. The pipeline supports PDF and text formats.

2. Define the Questions

   Create a text file or a list in the script with the questions you want to ask. Example:

   .. code-block:: python

      questions = [
          "What is the main objective of the paper?",
          "What methods were used in the research?",
          "What are the key findings?"
      ]

3. Run the Pipeline

   Execute the script with the input directory and the questions:

   .. code-block:: bash

      python ask_questions.py --input_dir /path/to/papers --questions_file questions.txt

4. View the Results

   The answers to each question for each paper will be saved in the output directory specified in the script. The default output format is CSV, but this can be customized.

Example Output
--------------

A sample CSV output might look like this:

+-----------------------------+------------------------------------------+------------------------------------------------------+
| Paper Title                 | Question                                 | Answer                                               |
+=============================+==========================================+======================================================+
| "Deep Learning in Medicine" | What is the main objective of the paper? | The main objective is to explore deep learning...     |
+-----------------------------+------------------------------------------+------------------------------------------------------+
| "AI and Ethics"             | What methods were used in the research?  | The research used a combination of qualitative...     |
+-----------------------------+------------------------------------------+------------------------------------------------------+

License
-------

This work is licensed under the Apache 2.0 open source license described in the `LICENSE` file in this repository.

Contact
-------

For any questions or support, please contact Rich Sharp at richpsharp@gmail.com.
