import streamlit as st
import pandas as pd
import re
import os
from groq import Groq
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
from io import StringIO
import hashlib

# === Initialize session state ===
def init_session_state():
    defaults = {
        'generated_code': "",
        'last_output': None,
        'last_error': "",
        'df': None,
        'code_history': [],
        'code_version': 0,
        'df_name': "df",
        'show_manual_edit': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# === Extract code from model response ===
def extract_code(response):
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    code_blocks = re.findall(r'```(?:python)?(.*?)```', clean_response, re.DOTALL)
    return '\n'.join([block.strip() for block in code_blocks]) if code_blocks else clean_response.strip()

# === Execute generated code safely ===
def execute_code(code, df):
    environment = {
        'pd': pd,
        'np': np,
        'plt': plt,
        st.session_state.df_name: df,
        'st': st,
        '__file__': 'generated_code.py'
    }
    
    old_stdout = sys.stdout
    try:
        plt.close('all')
        preamble = "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
        safe_code = re.sub(r"```python|```", '', code)  # Strip markdown
        safe_code = re.sub(r"pd\.read_csv\(.*?\)", st.session_state.df_name, safe_code)
        full_code = preamble + safe_code

        # Capture print output
        output_buffer = StringIO()
        sys.stdout = output_buffer

        # Execute the code
        exec(full_code, environment)

        # Restore stdout and get captured output
        sys.stdout = old_stdout
        captured_output = output_buffer.getvalue()
        
        return captured_output, None, environment

    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.format_exc()
        return None, tb, environment

# === Call Groq to generate code ===
def generate_code_with_groq(prompt, df):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

        data_context = f"""
Dataset Structure:
Columns: {', '.join(df.columns)}
First 3 rows:
{df.head(3).to_string(index=False)}
"""

        system_prompt = f"""You are a data science assistant. Generate Python code using:
- pandas, numpy, matplotlib, sklearn only
- DataFrame is already loaded as '{st.session_state.df_name}'
- Do not use file I/O operations
- For displaying results, use: st.write() for text/DataFrames, st.dataframe() for tables
- For plots: Always use fig, ax = plt.subplots() and st.pyplot(fig)
- Store final results in a variable called 'result' if needed
- Use print() statements for debugging output
- Wrap final code inside ```python``` blocks"""

        response = client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data_context},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"

# === Explain error in simple terms using AI ===
def explain_error_with_groq(error_text):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

        response = client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You are an expert Python error explainer. Convert this traceback into a short, clear explanation for another AI to fix."},
                {"role": "user", "content": error_text}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Could not explain error: {e})"

# === Main Streamlit App ===
def main():
    st.title("📊 AI Data Analyst Assistant")
    init_session_state()

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded!")
        st.dataframe(st.session_state.df.head())

    prompt = st.text_area("🔎 What would you like to analyze?", height=100)

    if st.button("✨ Generate Code", disabled=not uploaded_file or not prompt):
        with st.spinner("Generating code..."):
            generated = generate_code_with_groq(prompt, st.session_state.df)
            code = extract_code(generated)
            st.session_state.generated_code = code
            st.session_state.code_history.append(code)
            st.session_state.code_version += 1

    if st.session_state.generated_code:
        st.subheader(f"🧠 Generated Code (v{st.session_state.code_version})")
        st.code(st.session_state.generated_code, language='python')

        if st.button("🚀 Run Code"):
            with st.spinner("Executing..."):
                safe_code = st.session_state.generated_code
                # Disable potentially unsafe libraries
                for lib in ['xgboost', 'cudf', 'cupy', 'torch', 'tensorflow']:
                    safe_code = safe_code.replace(lib, f'# {lib} disabled')

                output, error, env = execute_code(safe_code, st.session_state.df)

                if error:
                    st.session_state.last_error = error
                    st.error("⚠️ Execution Error")
                    with st.expander("🔧 Full Traceback"):
                        st.code(error)
                else:
                    st.session_state.last_error = ""
                    st.success("✅ Code ran successfully!")
                    
                    # Display captured print output
                    if output and output.strip():
                        st.subheader("📝 Output:")
                        st.text(output)
                    
                    # Display any matplotlib figures
                    if plt.get_fignums():
                        st.subheader("📊 Plot:")
                        st.pyplot(plt.gcf())
                        plt.close('all')
                    
                    # Display any results stored in environment
                    for key in ['result', 'output', 'answer']:
                        if key in env and env[key] is not None:
                            st.subheader(f"🎯 {key.title()}:")
                            if isinstance(env[key], pd.DataFrame):
                                st.dataframe(env[key])
                            elif isinstance(env[key], (plt.Figure, matplotlib.figure.Figure)):
                                st.pyplot(env[key])
                            else:
                                st.write(env[key])

    # === Debug and Fix ===
    if st.session_state.last_error:
        st.subheader("🛠️ Debug Assistant")
        
        # Show error explanation
        with st.expander("🔧 Error Details", expanded=True):
            st.error("Current Error:")
            st.code(st.session_state.last_error)

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔁 Auto Fix Code"):
                with st.spinner("🔍 Understanding the error..."):
                    simplified_error = explain_error_with_groq(st.session_state.last_error)

                st.info("🧾 AI's Understanding:")
                st.write(simplified_error)

                # Enhanced fix prompt with more context
                fix_prompt = f"""
You are debugging Python code that failed. Here's the context:

DATASET INFO:
- DataFrame variable name: '{st.session_state.df_name}'
- Columns: {list(st.session_state.df.columns) if st.session_state.df is not None else 'Unknown'}
- Shape: {st.session_state.df.shape if st.session_state.df is not None else 'Unknown'}

ERROR ANALYSIS:
{simplified_error}

FULL ERROR TRACEBACK:
{st.session_state.last_error}

FAILED CODE:
```python
{st.session_state.generated_code}
```

INSTRUCTIONS:
1. Analyze the error carefully
2. Fix ONLY the specific issue causing the error
3. Use pandas, numpy, matplotlib, sklearn only
4. DataFrame is already loaded as '{st.session_state.df_name}'
5. For plots: use fig, ax = plt.subplots() then st.pyplot(fig)
6. For output: use st.write() or print()
7. DO NOT change the core functionality, only fix the bug
8. Return ONLY the corrected Python code in a ```python``` block

Generate the fixed code now:
"""

                with st.spinner("🛠️ Generating fix..."):
                    try:
                        fixed_response = generate_code_with_groq(fix_prompt, st.session_state.df)
                        fixed_code = extract_code(fixed_response)

                        # Compare codes
                        def normalize_code(code):
                            # Remove whitespace and comments for better comparison
                            normalized = re.sub(r'#.*', '', code)  # Remove comments
                            normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
                            return normalized.strip()

                        if normalize_code(fixed_code) != normalize_code(st.session_state.generated_code):
                            st.success("✅ AI generated a different fix!")
                            
                            # Show the fix
                            with st.expander("👀 Proposed Fix", expanded=True):
                                st.code(fixed_code, language='python')
                            
                            # Apply the fix
                            st.session_state.generated_code = fixed_code
                            st.session_state.code_history.append(fixed_code)
                            st.session_state.code_version += 1
                            st.session_state.last_error = ""
                            st.success("🎉 Code updated! Try running it again.")
                            st.rerun()
                        else:
                            st.warning("⚠️ AI returned the same code. The fix may not be obvious.")
                            st.info("💡 Try the manual edit option or rephrase your original request.")
                            
                    except Exception as e:
                        st.error(f"❌ Failed to generate fix: {str(e)}")
        
        with col2:
            if st.button("✏️ Manual Edit"):
                st.session_state.show_manual_edit = True
                st.rerun()
    
    # Manual editing interface
    if st.session_state.get('show_manual_edit', False):
        st.subheader("✏️ Manual Code Editor")
        
        edited_code = st.text_area(
            "Edit the code manually:",
            value=st.session_state.generated_code,
            height=300,
            key="manual_code_editor"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes"):
                st.session_state.generated_code = edited_code
                st.session_state.code_history.append(edited_code)
                st.session_state.code_version += 1
                st.session_state.last_error = ""
                st.session_state.show_manual_edit = False
                st.success("✅ Code updated manually!")
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel Edit"):
                st.session_state.show_manual_edit = False
                st.rerun()

if __name__ == "__main__":
    main()