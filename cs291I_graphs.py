import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify file paths
csv_path = 'data/recurrent-ppo-tray.csv'   # CSV file in the 'data' folder
save_path = 'plots/mean_reward_plot.png'    # Save the plot in the 'plots' folder

# Read the CSV file
df = pd.read_csv(csv_path)

# Optionally inspect the columns
print(df.columns)

# Filter to only include data with Step > 5
df_filtered = df[df['Step'] > 0]

# Set a publication-quality style using seaborn
sns.set_style('whitegrid')

# Create a Seaborn line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_filtered, x='Step', y='Value', label='Mean Reward')

# Add labels, title, and legend
plt.xlabel('Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Mean Reward per Step', fontsize=14)
plt.legend(loc='upper left')
# Optimize layout and save the figure
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()
