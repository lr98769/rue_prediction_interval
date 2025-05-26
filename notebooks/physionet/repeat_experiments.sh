for VARIABLE in 2024 2025 2026 2027
do
    echo "seed=$VARIABLE" > seed_file.py
    ipython --TerminalIPythonApp.file_to_run='2_mlp.ipynb'
done

