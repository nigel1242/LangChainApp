## Python Version:
!If your python is older or newer, please install python 3.11 from https://www.python.org/downloads/release/python-3110/
1. Open a new terminal
2. Run "python --version"
3. If you have Python 3.11, proceed to the next section.


## Setup:
If you have a GPU to run ollama models locally, follow the whole flow of this section.
Else, skip to "If you don't have a GPU"

1. Install ollama >> https://ollama.com/download/windows
2. Open terminal in VSCode
3. Pull ollama models and embedders: "ollama pull nomic-embed-text; ollama pull qwen2.5vl; ollama pull llama3:8b"
nomic-embed-text: This is for the ollama embedder.
qwen2.5vl: This is for multimodal mode, where you can drop images in.
llama3: This will be the main chat model.

If you don't have a GPU, you can only use openAI models.
Just enter your API key in the sidebar and select one of the GPT models.

1. Create virtual environment using "py -3.11 -m venv venv".
2. Run venv using "venv\scripts\activate"
3. Install requirements using "pip install -r requirements.txt"
4. Create .env file in project folder and copy paste the keys
5. Run app using "streamlit run app.py"

Quiz.py error: NOT FIXED 17/12
Sometimes showing hint will get this error
TypeError: join() argument must be str, bytes, or os.PathLike object, not 'NoneType'

File "C:\Users\kunji\Downloads\LangChainApp\venv\Lib\site-packages\streamlit\runtime\scriptrunner\exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
File "C:\Users\kunji\Downloads\LangChainApp\venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 667, in code_to_exec
    _mpa_v1(self._main_script_path)
File "C:\Users\kunji\Downloads\LangChainApp\venv\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 165, in _mpa_v1
    page.run()
File "C:\Users\kunji\Downloads\LangChainApp\venv\Lib\site-packages\streamlit\navigation\page.py", line 303, in run
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\kunji\Downloads\LangChainApp\pages\Quiz.py", line 217, in <module>
    entry = find_relevant_context_for_text(S["selected_subject"], img_query)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\kunji\Downloads\LangChainApp\modules\quizstats\file_utils.py", line 463, in find_relevant_context_for_text
    idx = _load_subject_index(subject_name)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\kunji\Downloads\LangChainApp\modules\quizstats\file_utils.py", line 413, in _load_subject_index
    root = ensure_subject_folders(subject_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\kunji\Downloads\LangChainApp\modules\quizstats\file_utils.py", line 40, in ensure_subject_folders
    root = os.path.join(SUBJECTS_DIR, subject_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "<frozen ntpath>", line 143, in join
File "<frozen genericpath>", line 152, in _check_arg_types