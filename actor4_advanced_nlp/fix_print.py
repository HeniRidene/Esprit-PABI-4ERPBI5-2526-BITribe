with open("runnable.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("runnable.py", "w", encoding="utf-8") as f:
    f.write("import sys\nimport io\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')\n")
    for line in lines:
        if "⚠️" in line:
            f.write(line.replace("⚠️", "[WARNING]"))
        else:
            f.write(line)
