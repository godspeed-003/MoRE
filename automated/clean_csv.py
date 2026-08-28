import os

CSV_PATH = r"c:\Users\Hp\Desktop\Waste\MoRE\sweep_results.csv"

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as f:
        lines = f.readlines()
    
    clean_lines = [lines[0]]
    for line in lines[1:]:
        if ",nan," not in line.lower():
            clean_lines.append(line)
            
    with open(CSV_PATH, "w") as f:
        f.writelines(clean_lines)
    print("CSV successfully cleaned!")
else:
    print("CSV not found.")
