best_color = "#88E7B8"
second_best_color = "#FAC05E"

def highlight_first_n_second_lowest(s, split=True):
    if split:
        s = s.map(lambda x: float(x.split(" ")[0]))
    smallest = s.nsmallest(1).iloc[-1] # Find the highest value
    second_smallest = s.nsmallest(2).iloc[-1]  # Find the second highest value
    output = []
    for v in s:
        if v == smallest:
            output.append(f'background-color: {best_color}; color: black')
        elif v == second_smallest:
            output.append(f'background-color: {second_best_color};  color: black')
        else:
            output.append("")
    return output

def highlight_first_n_second_highest(s, split=True):
    if split:
        s = s.map(lambda x: float(x.split(" ")[0]))
    highest = s.nlargest(1).iloc[-1] # Find the highest value
    second_highest = s.nlargest(2).iloc[-1]  # Find the second highest value
    output = []
    for v in s:
        if v == highest:
            output.append(f'background-color: {best_color}; color: black')
        elif v == second_highest:
            output.append(f'background-color: {second_best_color}; color: black')
        else:
            output.append("")
    return output