import csv
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from prefixspan import PrefixSpan
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class DataMiner:
    def __init__(self, data_file):
        self.data_file = data_file

    def load_data(self):
        with open(self.data_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        df = pd.DataFrame(rows)

        transactions = {}
        departments = {}
        for index, row in df.iterrows():
            for item in row:
                if item and len(item.split()) == 3:
                    department, time, price = item.split()
                    if index not in transactions:
                        transactions[index] = {
                            "departments": [],
                            "times": [],
                            "prices": [],
                        }
                    if department not in departments:
                        departments[department] = {"times": [], "prices": []}

                    transactions[index]["departments"].append(department)
                    transactions[index]["times"].append(float(time))
                    transactions[index]["prices"].append(float(price))

                    departments[department]["times"].append(float(time))
                    departments[department]["prices"].append(float(price))

        return transactions, departments

    def load_extra(self, test_file):
        with open(test_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        df = pd.DataFrame(rows)

        transactions = {}
        departments = {}
        for index, row in df.iterrows():
            for item in row:
                if item and len(item.split()) == 3:
                    department, time, price = item.split()
                    if index not in transactions:
                        transactions[index] = {
                            "departments": [],
                            "times": [],
                            "prices": [],
                        }
                    if department not in departments:
                        departments[department] = {"times": [], "prices": []}

                    transactions[index]["departments"].append(department)
                    transactions[index]["times"].append(float(time))
                    transactions[index]["prices"].append(float(price))

                    departments[department]["times"].append(float(time))
                    departments[department]["prices"].append(float(price))

        return transactions, departments

    def mine_association_rules(self, support, lift):
        try:
            with open("case I/association_rules.pkl", "rb") as f:
                df_ar = pickle.load(f)
                print("Association rules read from file.")
        except FileNotFoundError:
            transactions = self.load_data()
            encoder = TransactionEncoder()
            association_data = encoder.fit_transform(transactions)
            association_df = pd.DataFrame(association_data, columns=encoder.columns_)
            association_df = apriori(
                association_df, min_support=0.001, use_colnames=True, low_memory=True
            )
            df_ar = association_rules(
                association_df, metric="confidence", min_threshold=0.75
            )
            with open("case I/association_rules.pkl", "wb") as f:
                pickle.dump(df_ar, f)

        association_df_filtered = df_ar[
            (df_ar["support"] >= support)
            & (df_ar["lift"] >= lift)
            & (df_ar["antecedents"].apply(lambda x: len(x)) > 1)
        ]

        sorted_rules = association_df_filtered.sort_values(
            by=["support", "confidence", "lift"], ascending=[False, False, False]
        )

        return sorted_rules

    def mine_sequential_rules(self, threshold):
        try:
            with open("case I/sequential_rules.pkl", "rb") as f:
                frequent_sequences = pickle.load(f)
                print("Sequential rules read from file.")
        except FileNotFoundError:
            transactions = self.load_data()
            frequent_sequences = PrefixSpan(transactions)
            frequent_sequences = list(frequent_sequences.frequent(1, closed=True))
            with open("case I/sequential_rules.pkl", "wb") as f:
                pickle.dump(frequent_sequences, f)

        filtered_sequences = [
            seq
            for support, seq in frequent_sequences
            if support >= threshold and len(seq) > 1
        ]

        return filtered_sequences

    def find_time_outliers(
        self, lower_percentile, upper_percentile, k, threshold_percentile
    ):
        transactions, departments = self.load_data()
        time_bounds = {}
        total_time_spent = {}
        total_items = {}

        for department, data in departments.items():
            time_values = data["times"]
            time = np.array(time_values)

            lower_bound = float(np.percentile(time, lower_percentile))
            upper_bound = float(np.percentile(time, upper_percentile))

            time_bounds[department] = (lower_bound, upper_bound)

        for transaction, data in transactions.items():
            departments = data["departments"]
            price_values = data["times"]
            total_time_spent[transaction] = sum(price_values)
            total_items[transaction] = len(departments)

        X = np.array(
            [
                [total_time_spent[transaction], total_items[transaction]]
                for transaction in transactions
            ]
        )

        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        clusters = {i: [] for i in range(len(cluster_centroids))}

        distances = pairwise_distances_argmin_min(X, cluster_centroids)[1]

        threshold = np.percentile(distances, threshold_percentile)

        # Assign transactions to clusters
        for transaction, _ in total_time_spent.items():
            cluster_label = cluster_labels[transaction]
            clusters[cluster_label].append(transaction)

        plt.figure(figsize=(10, 6))
        for label, transactions in clusters.items():
            plt.scatter(
                [total_time_spent[t] for t in transactions],
                [total_items[t] for t in transactions],
                label=f"Cluster {label}",
                alpha=0.5,
            )
        plt.scatter(
            cluster_centroids[:, 0],
            cluster_centroids[:, 1],
            s=300,
            c="red",
            marker="X",
            edgecolors="black",
            label="Centroids",
        )
        plt.title("Clusters and Centroids")
        plt.xlabel("Total Time Spent")
        plt.ylabel("Total Items")
        plt.legend()
        plt.grid(True)

        print("Time bounds and centroids determined.")

        return time_bounds, cluster_centroids, threshold

    def find_price_outliers(self, k, threshold_percentile):
        transactions, departments = self.load_data()
        total_price_spent = {}
        total_items = {}

        # TRANSACTION-LEVEL CLUSTERING BASED ON TOTAL SPENT AND TOTAL ITEMS
        for transaction, data in transactions.items():
            departments = data["departments"]
            price_values = data["prices"]
            total_price_spent[transaction] = sum(price_values)
            total_items[transaction] = len(departments)

        X = np.array(
            [
                [total_price_spent[transaction], total_items[transaction]]
                for transaction in transactions
            ]
        )

        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(X)

        cluster_labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        clusters = {i: [] for i in range(len(cluster_centroids))}

        distances = pairwise_distances_argmin_min(X, cluster_centroids)[1]

        threshold = np.percentile(distances, threshold_percentile)

        # Assign transactions to clusters
        for transaction, _ in total_price_spent.items():
            cluster_label = cluster_labels[transaction]
            clusters[cluster_label].append(transaction)

        plt.figure(figsize=(10, 6))
        for label, transactions in clusters.items():
            plt.scatter(
                [total_price_spent[t] for t in transactions],
                [total_items[t] for t in transactions],
                label=f"Cluster {label}",
                alpha=0.5,
            )
        plt.scatter(
            cluster_centroids[:, 0],
            cluster_centroids[:, 1],
            s=300,
            c="red",
            marker="X",
            edgecolors="black",
            label="Centroids",
        )
        plt.title("Clusters and Centroids")
        plt.xlabel("Total Price Spent")
        plt.ylabel("Total Items")
        plt.legend()
        plt.grid(True)

        print("Price centroids determined.")

        return cluster_centroids, threshold

    def department_clusters(self, predict_department_df):
        training_transactions, training_departments = self.load_data()
        test_transactions, test_departments = self.load_extra(
            "case I/supermarket_extra.csv"
        )
        labels_df = pd.read_csv("case I/supermarket_extra_labels.csv", header=None)

        training_department_counts = {department: [] for department in range(1, 19)}
        for transaction, data in training_transactions.items():
            departments_visited = data["departments"]
            visit_count = {department: 0 for department in range(1, 19)}
            for department in departments_visited:
                department = int(department)
                visit_count[department] += 1
            for department, count in visit_count.items():
                training_department_counts[department].append(count)

        training_department_df = pd.DataFrame(training_department_counts)

        test_department_counts = {department: [] for department in range(1, 19)}
        for transaction, data in test_transactions.items():
            departments_visited = data["departments"]
            visit_count = {department: 0 for department in range(1, 19)}
            for department in departments_visited:
                department = int(department)
                visit_count[department] += 1
            for department, count in visit_count.items():
                test_department_counts[department].append(count)

        test_department_df = pd.DataFrame(test_department_counts)

        merged_df = pd.concat([test_department_df, labels_df], axis=1)
        merged_df.rename(columns={0: "Amount_Stolen"}, inplace=True)

        non_fraud_df = merged_df[merged_df["Amount_Stolen"] == 0].copy()
        fraud_df = merged_df[merged_df["Amount_Stolen"] > 0].copy()

        non_fraud_df.drop(columns=["Amount_Stolen"], inplace=True)
        fraud_df.drop(columns=["Amount_Stolen"], inplace=True)

        cluster_range = range(1, 11)
        inertias = []

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            kmeans.fit(training_department_df)

            inertias.append(kmeans.inertia_)

        plt.plot(cluster_range, inertias, marker="o")
        plt.title("Elbow Method for Optimal K (Before Median)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.xticks(cluster_range)
        plt.grid(True)
        # plt.show()

        kmeans = KMeans(n_clusters=6, random_state=1)
        cluster_labels = kmeans.fit_predict(training_department_df)
        training_department_df["Cluster"] = cluster_labels

        training_cluster_counts = (
            training_department_df["Cluster"].value_counts().sort_index()
        )

        plt.figure(figsize=(8, 6))
        plt.bar(
            training_cluster_counts.index,
            training_cluster_counts.values,
            color="orange",
        )
        plt.title("Distribution of Transactions among Training Data Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(training_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        non_fraud_cluster_labels = kmeans.predict(non_fraud_df)
        non_fraud_df["Predicted_Cluster"] = non_fraud_cluster_labels
        non_fraud_cluster_counts = (
            non_fraud_df["Predicted_Cluster"].value_counts().sort_index()
        )

        print(non_fraud_cluster_counts / non_fraud_cluster_counts.sum())

        plt.figure(figsize=(8, 6))
        plt.bar(
            non_fraud_cluster_counts.index,
            non_fraud_cluster_counts.values,
            color="orange",
        )
        plt.title("Distribution of Transactions among Non-Fraud Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(non_fraud_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        fraud_cluster_labels = kmeans.predict(fraud_df)
        fraud_df["Predicted_Cluster"] = fraud_cluster_labels
        fraud_cluster_counts = fraud_df["Predicted_Cluster"].value_counts().sort_index()

        print(fraud_cluster_counts / fraud_cluster_counts.sum())

        plt.figure(figsize=(8, 6))
        plt.bar(fraud_cluster_counts.index, fraud_cluster_counts.values, color="orange")
        plt.title("Distribution of Transactions among Fraud Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(fraud_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        predict_cluster_labels = kmeans.predict(predict_department_df)
        predict_department_df["Predicted_Cluster"] = predict_cluster_labels

        fraud_cluster_transactions = predict_department_df[
            predict_department_df["Predicted_Cluster"] == 1
        ]
        fraud_cluster_transaction_ids = fraud_cluster_transactions.index.tolist()

        return fraud_cluster_transaction_ids

    def time_clusters(self, predict_department_df):
        training_transactions, training_departments = self.load_data()
        test_transactions, test_departments = self.load_extra(
            "case I/supermarket_extra.csv"
        )
        labels_df = pd.read_csv("case I/supermarket_extra_labels.csv", header=None)

        training_department_times = {department: [] for department in range(1, 19)}
        for transaction, data in training_transactions.items():
            departments_visited = data["departments"]
            times_spent = data["times"]
            time_spent_per_department = {department: 0 for department in range(1, 19)}
            for department, time_spent in zip(departments_visited, times_spent):
                department = int(department)
                time_spent_per_department[department] += time_spent
            for department, time_spent in time_spent_per_department.items():
                training_department_times[department].append(time_spent)

        training_department_times_df = pd.DataFrame(training_department_times)

        test_department_times = {department: [] for department in range(1, 19)}
        for transaction, data in test_transactions.items():
            departments_visited = data["departments"]
            times_spent = data["times"]
            time_spent_per_department = {department: 0 for department in range(1, 19)}
            for department, time_spent in zip(departments_visited, times_spent):
                department = int(department)
                time_spent_per_department[department] += time_spent
            for department, time_spent in time_spent_per_department.items():
                test_department_times[department].append(time_spent)

        test_department_times_df = pd.DataFrame(test_department_times)

        merged_df = pd.concat([test_department_times_df, labels_df], axis=1)
        merged_df.rename(columns={0: "Amount_Stolen"}, inplace=True)

        non_fraud_df = merged_df[merged_df["Amount_Stolen"] == 0].copy()
        fraud_df = merged_df[merged_df["Amount_Stolen"] > 0].copy()

        non_fraud_df.drop(columns=["Amount_Stolen"], inplace=True)
        fraud_df.drop(columns=["Amount_Stolen"], inplace=True)

        cluster_range = range(1, 11)
        inertias = []

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            kmeans.fit(training_department_times_df)

            inertias.append(kmeans.inertia_)

        plt.plot(cluster_range, inertias, marker="o")
        plt.title("Elbow Method for Optimal K (Before Median)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.xticks(cluster_range)
        plt.grid(True)
        # plt.show()

        kmeans = KMeans(n_clusters=6, random_state=1)
        cluster_labels = kmeans.fit_predict(training_department_times_df)
        training_department_times_df["Cluster"] = cluster_labels

        training_cluster_counts = (
            training_department_times_df["Cluster"].value_counts().sort_index()
        )

        plt.figure(figsize=(8, 6))
        plt.bar(
            training_cluster_counts.index,
            training_cluster_counts.values,
            color="orange",
        )
        plt.title("Distribution of Transactions among Training Data Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(training_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        non_fraud_cluster_labels = kmeans.predict(non_fraud_df)
        non_fraud_df["Predicted_Cluster"] = non_fraud_cluster_labels
        non_fraud_cluster_counts = (
            non_fraud_df["Predicted_Cluster"].value_counts().sort_index()
        )

        print(non_fraud_cluster_counts / non_fraud_cluster_counts.sum())

        plt.figure(figsize=(8, 6))
        plt.bar(
            non_fraud_cluster_counts.index,
            non_fraud_cluster_counts.values,
            color="orange",
        )
        plt.title("Distribution of Transactions among Non-Fraud Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(non_fraud_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        fraud_cluster_labels = kmeans.predict(fraud_df)
        fraud_df["Predicted_Cluster"] = fraud_cluster_labels
        fraud_cluster_counts = fraud_df["Predicted_Cluster"].value_counts().sort_index()

        print(fraud_cluster_counts / fraud_cluster_counts.sum())

        plt.figure(figsize=(8, 6))
        plt.bar(fraud_cluster_counts.index, fraud_cluster_counts.values, color="orange")
        plt.title("Distribution of Transactions among Fraud Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(fraud_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        predict_cluster_labels = kmeans.predict(predict_department_df)
        predict_department_df["Predicted_Cluster"] = predict_cluster_labels

        fraud_cluster_transactions = predict_department_df[
            predict_department_df["Predicted_Cluster"] == 1
        ]
        fraud_cluster_transaction_ids = fraud_cluster_transactions.index.tolist()

        return fraud_cluster_transaction_ids

    def spending_clusters(self, predict_department_df):
        training_transactions, training_departments = self.load_data()
        test_transactions, test_departments = self.load_extra(
            "case I/supermarket_extra.csv"
        )
        labels_df = pd.read_csv("case I/supermarket_extra_labels.csv", header=None)

        training_department_spendings = {department: [] for department in range(1, 19)}
        for transaction, data in training_transactions.items():
            departments_visited = data["departments"]
            money_spent = data["prices"]
            money_spent_per_department = {department: 0 for department in range(1, 19)}
            for department, money_spent in zip(departments_visited, money_spent):
                department = int(department)
                money_spent_per_department[department] += money_spent
            for department, money_spent in money_spent_per_department.items():
                training_department_spendings[department].append(money_spent)

        training_department_spendings_df = pd.DataFrame(training_department_spendings)

        test_department_spendings = {department: [] for department in range(1, 19)}
        for transaction, data in test_transactions.items():
            departments_visited = data["departments"]
            money_spent = data["prices"]
            money_spent_per_department = {department: 0 for department in range(1, 19)}
            for department, money_spent in zip(departments_visited, money_spent):
                department = int(department)
                money_spent_per_department[department] += money_spent
            for department, money_spent in money_spent_per_department.items():
                test_department_spendings[department].append(money_spent)

        test_department_spendings_df = pd.DataFrame(test_department_spendings)

        merged_df = pd.concat([test_department_spendings_df, labels_df], axis=1)
        merged_df.rename(columns={0: "Amount_Stolen"}, inplace=True)

        non_fraud_df = merged_df[merged_df["Amount_Stolen"] == 0].copy()
        fraud_df = merged_df[merged_df["Amount_Stolen"] > 0].copy()

        non_fraud_df.drop(columns=["Amount_Stolen"], inplace=True)
        fraud_df.drop(columns=["Amount_Stolen"], inplace=True)

        cluster_range = range(1, 11)
        inertias = []

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=1)
            kmeans.fit(training_department_spendings_df)

            inertias.append(kmeans.inertia_)

        plt.plot(cluster_range, inertias, marker="o")
        plt.title("Elbow Method for Optimal K (Before Median)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")
        plt.xticks(cluster_range)
        plt.grid(True)
        # plt.show()

        kmeans = KMeans(n_clusters=6, random_state=1)
        cluster_labels = kmeans.fit_predict(training_department_spendings_df)
        training_department_spendings_df["Cluster"] = cluster_labels

        training_cluster_counts = (
            training_department_spendings_df["Cluster"].value_counts().sort_index()
        )

        plt.figure(figsize=(8, 6))
        plt.bar(
            training_cluster_counts.index,
            training_cluster_counts.values,
            color="orange",
        )
        plt.title("Distribution of Transactions among Training Data Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(training_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        non_fraud_cluster_labels = kmeans.predict(non_fraud_df)
        non_fraud_df["Predicted_Cluster"] = non_fraud_cluster_labels
        non_fraud_cluster_counts = (
            non_fraud_df["Predicted_Cluster"].value_counts().sort_index()
        )

        print(non_fraud_cluster_counts / non_fraud_cluster_counts.sum())

        plt.figure(figsize=(8, 6))
        plt.bar(
            non_fraud_cluster_counts.index,
            non_fraud_cluster_counts.values,
            color="orange",
        )
        plt.title("Distribution of Transactions among Non-Fraud Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(non_fraud_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        fraud_cluster_labels = kmeans.predict(fraud_df)
        fraud_df["Predicted_Cluster"] = fraud_cluster_labels
        fraud_cluster_counts = fraud_df["Predicted_Cluster"].value_counts().sort_index()

        print(fraud_cluster_counts / fraud_cluster_counts.sum())

        plt.figure(figsize=(8, 6))
        plt.bar(fraud_cluster_counts.index, fraud_cluster_counts.values, color="orange")
        plt.title("Distribution of Transactions among Fraud Clusters")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Transactions")
        plt.xticks(fraud_cluster_counts.index)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        # plt.show()

        predict_cluster_labels = kmeans.predict(predict_department_df)
        predict_department_df["Predicted_Cluster"] = predict_cluster_labels

        fraud_cluster_transactions = predict_department_df[
            predict_department_df["Predicted_Cluster"] == 1
        ]
        fraud_cluster_transaction_ids = fraud_cluster_transactions.index.tolist()

        return fraud_cluster_transaction_ids
