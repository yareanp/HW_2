import data

if __name__ == "__main__":
    partA()
    partB()

def partA():
    print("Part A: ")
    df = data.load_data("london.csv")
    df = data.add_new_columns(df)
    data.data_analysis(df)
    print()

def partB():
    pass