import pandas as pd
import argparse
import os
import sys

import html
def preprocess_string(s):
    if isinstance(s, int):
        s = str(s)
    if pd.isna(s):  # Handle NaN values
        return ""
    s = html.escape(s)  # Escape special HTML characters
    s = s.replace("\n", "<br>")  # Replace newlines with <br>
    s = s.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")  # Replace tabs with spaces
    return s
seeds = [0, 1, 2, 3, 4, 5, 6, 7]

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Responses Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; color: #333; }
        .page { display: none; }
        .page.active { display: block; }
        .container { display: flex; border: 1px solid #ccc; margin: 10px; padding: 10px; }
        .render, .description { flex: 1; margin: 10px; }
        .hidden { display: none; margin-top: 10px; line-height: 1.6; }
        button { margin-top: 10px; }
        pre { 
        background: #000000;
        color: #f5f5f5;
        padding: 10px; 
        border-radius: 5px; 
        white-space: pre-wrap; 
        word-wrap: break-word; 
        text-align: center; 
        margin: 0 auto;
        font-weight: bold;
        font-size: 1.2em;
        }
    </style>
    <script>
        let currentPage = 0;
        let currentSeed = 0;

        function showPage(pageIndex) {
            const pages = document.querySelectorAll(`.seed-${currentSeed} .page`);
            pages.forEach((page, index) => {
                page.classList.toggle('active', index === pageIndex);
            });
            currentPage = pageIndex;
            const stepSelect = document.getElementById('step-select');
            stepSelect.value = pageIndex;
        }

        function nextPage() {
            const pages = document.querySelectorAll(`.seed-${currentSeed} .page`);
            console.log(pages.length)
            if (currentPage < pages.length - 1) {
                showPage(currentPage + 1);
            }
        }

        function prevPage() {
            if (currentPage > 0) {
                showPage(currentPage - 1);
            }
        }

        function switchSeed(seedIndex) {
            const allSeeds = document.querySelectorAll('.seed-container');
            allSeeds.forEach((seed, index) => {
                seed.style.display = index === seedIndex ? 'block' : 'none';
            });
            currentSeed = seedIndex;
            currentPage = 0;
            showPage(0);

            // Update step dropdown for the new seed
            const stepSelect = document.getElementById('step-select');
            stepSelect.innerHTML = '';
            const pages = document.querySelectorAll(`.seed-${currentSeed} .page`);
            pages.forEach((_, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `Step ${index}`;
                stepSelect.appendChild(option);
            });
        }

        function switchStep(stepIndex) {
            showPage(stepIndex);
        }

        function toggleVisibility(button) {
            const content = button.nextElementSibling;
            if (content.style.display === "none" || content.style.display === "") {
                content.style.display = "block";
            } else {
                content.style.display = "none";
            }
        }
        window.onload = function () {
            switchSeed(0);
        };
    </script>
</head>
<body>
    <h1>LLM Responses Visualization</h1>
    <label for="seed-select">Select Seed:</label>
    <select id="seed-select" onchange="switchSeed(this.selectedIndex)">
"""
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
parser = argparse.ArgumentParser(description='Visualize LLM responses from CSV files.')
parser.add_argument('--f', type=str, default=None, help='Path to the CSV file')
args = parser.parse_args()
args.f = args.f.replace("0.csv", "seed.csv")
if "freeway" in args.f:
    from envs.freeway import seed_mapping
    difficulty = 'E' if 'E' in args.f else 'M' if 'M' in args.f else 'H' if 'H' in args.f else 'M'
    seed_mapping = seed_mapping[difficulty]
elif "snake" in args.f:
    seed_mapping = {0: 1000, 1: 1001, 2: 1002, 3: 1003, 4: 1004, 5: 1005, 6: 1006, 7: 1007}
elif "overcooked" in args.f:
    seed_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
# Add seed options to the dropdown
for seed in seeds:
    html_content += f'<option value="{seed}">Seed {seed_mapping[seed]}</option>'
html_content += """
    </select>
    <label for="step-select">Select Step:</label>
    <select id="step-select" onchange="switchStep(this.selectedIndex)">
    </select>
    <button onclick="prevPage()">Previous</button>
    <button onclick="nextPage()">Next</button>
"""

for seed in seeds:
    csv_path = args.f.replace("seed", str(seed))
    print(csv_path)
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)

    html_content += f'<div class="seed-container seed-{seed}" style="display: {"block" if seed == 0 else "none"};">'
    page_index = 0
    plan = ""

    for _, row in df.iterrows():
        render = row["render"]
        plan = row["scratch_pad"] if "scratch_pad" in row else row.get("plan", "Not Recorded")
        action = row["selected_action"] if "selected_action" in row else row['action']
        follow_plan = row["follow_plan"] if "follow_plan" in row else "Not Recorded"
        reward = row["reward"] if "reward" in row else "Not Recorded"
        meta_control = row["meta_control"] if "meta_control" in row else "Not Recorded"
        dropped_items = ["render", "action", "follow_plan", 'reward', 'plan', 'meta_control', "Unnamed: 0"]
        dropped_items = [item for item in dropped_items if item in row]
        other_columns = {key: preprocess_string(value) for key, value in row.drop(dropped_items).items()}
        other_columns_html = "".join(
            [f"<button onclick='toggleVisibility(this)'>{key}</button><div class='hidden' style='max-width: 800px; margin: 0 auto;'><p>{value}</p></div>"
            for key, value in other_columns.items()]
        )

        html_content += f"""
        <div class="page {'active' if page_index == 0 else ''}">
            <div class="container">
                <div class="render">
                    <h3>Render:</h3>
                    <pre>{render}</pre>
                </div>
                <div class="description">
                    <h3>Meta Control:</h3>
                    <p>{meta_control}</p>
                    <h3>Belief State:</h3>
                    <p>{plan}</p>
                    <h3>Follow Plan:</h3>
                    <p>{follow_plan}</p>
                    <h3>Selected Action | Total Reward</h3>
                    <p>{action} | {reward}</p>
                </div>
            </div>
            <div>
                {other_columns_html}
            </div>
        </div>
        """
        page_index += 1

    html_content += "</div>"

html_content += """
</body>
</html>
"""

# output_path = "index.html"
output_path = args.f.replace("_seed.csv", "_visualization.html")
if os.path.exists(output_path):
    os.remove(output_path)
output_path = os.path.join(os.path.dirname(args.f), os.path.dirname(args.f).split('/')[-1] + ".html")

with open(output_path, "w") as f:
    f.write(html_content)

print(f"Visualization saved to {output_path}. Open it in your browser to view.")
