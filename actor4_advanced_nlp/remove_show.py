with open("runnable.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("runnable.py", "w", encoding="utf-8") as f:
    for line in lines:
        if "plt.show()" in line:
            f.write(line.replace("plt.show()", "#plt.show()"))
        else:
            f.write(line)
