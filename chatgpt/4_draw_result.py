import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = [r'$QMST_{nll}$', r'$QMST_{nll}$+GA', r'$ChatGPT$', r'$ChatGPT+GA$']
counts = [21.0, 22.5, 95.0, 98.0]
bar_labels = ['peachpuff', 'goldenrod', 'skyblue', 'steelblue']
bar_colors = ['peachpuff', 'goldenrod', 'skyblue', 'steelblue']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('Win rate (%)')
ax.set_title('Win rate against gold label, judged by GPT-4')
for index, value in enumerate(counts):
    plt.text(index-0.1,value+0.5,value)
plt.axhline(y=50, color='tomato', linestyle='dashed')
plt.savefig("./Figure_1.png")
plt.show()
