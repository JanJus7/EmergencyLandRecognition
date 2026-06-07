import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/training_history_V3.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(df['Epoch'], df['Train_Loss'], label='Strata Treningowa', color='red', linestyle='--', linewidth=2)
ax1.plot(df['Epoch'], df['Val_Loss'], label='Strata Walidacyjna', color='darkred', linewidth=2)
ax1.set_xlabel('Epoka', fontsize=12)
ax1.set_ylabel('CrossEntropyLoss', fontsize=12)
ax1.set_title('Loss', fontsize=14)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(fontsize=12)

ax2.plot(df['Epoch'], df['Val_Accuracy'], label='Dokładność Walidacyjna', color='blue', linewidth=2)
ax2.set_xlabel('Epoka', fontsize=12)
ax2.set_ylabel('Skuteczność klasyfikacji', fontsize=12)
ax2.set_title('Accuracy', fontsize=14)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig('learning_curves_clean.pdf', dpi=300)