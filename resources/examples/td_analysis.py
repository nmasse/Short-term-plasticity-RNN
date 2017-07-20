with open('overlap_set.txt', 'r') as f1:
    with open('set_list250_8.txt', 'r') as f2:
        overlap = set(f1).intersection(f2)

overlap.discard('\n')

with open('overlap_set2.txt', 'w') as f:
    for line in overlap:
        f.write(line)
