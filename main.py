import clustering
import data


def partA():
    print("Part A: ")
    df = data.load_data("london.csv")
    df = data.add_new_columns(df)
    data.data_analysis(df)
    print()

def partB():
    print("Part B: ")
    df = data.load_data("london.csv")
    transformed_df = clustering.transform_data(df, ["cnt", "t1"])
    for k in (2,3,5):
        print(f"k = {k}")
        labels, centroids = clustering.kmeans(transformed_df, k)
        print(np.array_str(centroids, precision=3, suppress_small=True))
        if k != 5:
            print()
        clustering.visualize_results(transformed_df, labels, centroids, f"kMeansResult{k}.png")


if __name__ == "__main__":
    partA()
    partB()

