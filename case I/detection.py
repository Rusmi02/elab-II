import csv
import random
import numpy as np
import pandas as pd
from project import DataMiner
from scipy import stats
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import seaborn as sns


class FraudDetector:
    def __init__(self, data_miner):
        self.data_miner = data_miner

    def load_transactions(self, filepath):
        with open(filepath, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            test_transactions = []
            transaction_ids = []
            for row in reader:
                transaction = []
                transaction_info = row[0].split(":")
                transaction_index = transaction_info[0]
                first_transaction = transaction_info[1]
                if first_transaction:
                    department, time, price = first_transaction.split()
                    transaction.append((department, float(time), float(price)))
                for item in row:
                    if item and len(item.split()) == 3:
                        department, time, price = item.split()
                        transaction.append((department, float(time), float(price)))
                if transaction:
                    test_transactions.append(transaction)
                    transaction_ids.append(transaction_index)

            return test_transactions, transaction_ids

    def detect_fraud(self, test_transactions, transaction_ids, score_threshold):
        flagged_transactions = []

        # CLUSTERING BASED ON DEPARTMENTS VISITED
        prediction_department_counts = {department: [] for department in range(1, 19)}
        for idx, transaction in zip(transaction_ids, test_transactions):
            visit_count = {department: 0 for department in range(1, 19)}
            for department, _, _ in transaction:
                department = int(department)
                visit_count[department] += 1
            for department, count in visit_count.items():
                prediction_department_counts[department].append(count)

        prediction_department_df = pd.DataFrame(
            prediction_department_counts, index=transaction_ids
        )

        fraud_cluster_ids = data_miner.department_clusters(prediction_department_df)
        print("CLUSTERING BASED ON DEPARTMENTS VISITED")
        if len(fraud_cluster_ids) < 150:
            print(f"{len(fraud_cluster_ids)}")
            for transaction_id in fraud_cluster_ids:
                transaction_number = transaction_id.split()[1]
                print(f"{transaction_number}")
        elif len(fraud_cluster_ids) > 150:
            random_selected_transactions = random.sample(fraud_cluster_ids, 150)
            print(f"{len(random_selected_transactions)} / {len(fraud_cluster_ids)}")
            for transaction_id in random_selected_transactions:
                transaction_number = transaction_id.split()[1]
                print(f"{transaction_number}")

        # CLUSTERING BASED ON TIME SPENT AT DEPARTMENTS
        prediction_department_times = {department: [] for department in range(1, 19)}
        for idx, transaction in zip(transaction_ids, test_transactions):
            time_spent_per_department = {department: 0 for department in range(1, 19)}
            for department, time_spent, _ in transaction:
                department = int(department)
                time_spent_per_department[department] += time_spent
            for department, time_spent in time_spent_per_department.items():
                prediction_department_times[department].append(time_spent)

        prediction_department_times_df = pd.DataFrame(
            prediction_department_times, index=transaction_ids
        )

        fraud_cluster_ids = data_miner.time_clusters(prediction_department_times_df)
        print("CLUSTERING BASED ON TIME SPENT AT DEPARTMENTS")
        if len(fraud_cluster_ids) < 150:
            print(f"{len(fraud_cluster_ids)}")
            for transaction_id in fraud_cluster_ids:
                transaction_number = transaction_id.split()[1]
                print(f"{transaction_number}")
        elif len(fraud_cluster_ids) > 150:
            random_selected_transactions = random.sample(fraud_cluster_ids, 150)
            print(f"{len(random_selected_transactions)} / {len(fraud_cluster_ids)}")
            for transaction_id in random_selected_transactions:
                transaction_number = transaction_id.split()[1]
                print(f"{transaction_number}")

        # CLUSTERING BASED ON MONEY SPENT AT DEPARTMENTS
        prediction_department_spendings = {department: [] for department in range(1, 19)}
        for idx, transaction in zip(transaction_ids, test_transactions):
            money_spent_per_department = {department: 0 for department in range(1, 19)}
            for department, _, money_spent in transaction:
                department = int(department)
                money_spent_per_department[department] += money_spent
            for department, money_spent in money_spent_per_department.items():
                prediction_department_spendings[department].append(money_spent)

        prediction_department_spendings_df = pd.DataFrame(
            prediction_department_spendings, index=transaction_ids
        )

        fraud_cluster_ids = data_miner.spending_clusters(prediction_department_spendings_df)
        print("CLUSTERING BASED ON MONEY SPENT AT DEPARTMENTS")
        if len(fraud_cluster_ids) < 150:
            print(f"{len(fraud_cluster_ids)}")
            for transaction_id in fraud_cluster_ids:
                transaction_number = transaction_id.split()[1]
                print(f"{transaction_number}")
        elif len(fraud_cluster_ids) > 150:
            random_selected_transactions = random.sample(fraud_cluster_ids, 150)
            print(f"{len(random_selected_transactions)} / {len(fraud_cluster_ids)}")
            for transaction_id in random_selected_transactions:
                transaction_number = transaction_id.split()[1]
                print(f"{transaction_number}")

        # MINE ASSOCIATION RULES
        sorted_rules = data_miner.mine_association_rules(support=0.2, lift=1.5)
        print(sorted_rules)
        # MINE SEQUENTIAL RULES
        frequent_sequences = data_miner.mine_sequential_rules(threshold=460)
        print(frequent_sequences)
        # FIND TIME OUTLIERS
        time_bounds, time_cluster_centroids, time_threshold = (
            data_miner.find_time_outliers(
                lower_percentile=5, upper_percentile=95, k=4, threshold_percentile=83.5
            )
        )
        print(time_bounds)
        print(time_cluster_centroids)
        # FIND PRICE OUTLIERS
        price_cluster_centroids, price_threshold = data_miner.find_price_outliers(
            k=4, threshold_percentile=83.5
        )
        print(price_cluster_centroids)

        total_ar_score = 0
        total_seq_score = 0
        total_time_score = 0
        total_price_score = 0
        for idx, transaction in zip(transaction_ids, test_transactions):
            # STARTING SCORE
            fraud_score = 0

            # CHECK ASSOCIATION RULES (increase in support = less flagged transactions)
            for _, rule in sorted_rules.iterrows():
                antecedent = set(rule["antecedents"])
                consequent = set(rule["consequents"])
                # if we find antecedent but not consequent, increase fraud score
                if antecedent.issubset(
                    set(item[0] for item in transaction)
                ) and not consequent.issubset(set(item[0] for item in transaction)):
                    fraud_score += 1.0
                    total_ar_score += 1.0
                    break

            # CHECK SEQUENTIAL RULES (decrease in threshold = less flagged transactions)
            no_sequences_found = True
            for seq in frequent_sequences:
                for start_index in range(len(transaction) - len(seq) + 1):
                    end_index = start_index + len(seq)
                    sub_transaction = transaction[start_index:end_index]
                    if [item[0] for item in sub_transaction] == seq:
                        no_sequences_found = False
                        break

            # if no matching sequences are found, increase fraud score
            if no_sequences_found:
                fraud_score += 1.0
                total_seq_score += 1.0

            # CHECK TIME OUTLIERS (stricter bounds = less flagged transactions)
            count_time_exceeded = 0
            for department, time, _ in transaction:
                lower_bound, upper_bound = time_bounds.get(department, (None, None))
                # if a time is outside the bounds, increase fraud score
                if time < lower_bound or time > upper_bound:
                    count_time_exceeded += 1

            if count_time_exceeded > 2:
                fraud_score += 0.0
                total_time_score += 0.0

            # CHECK TOTAL TIME SPENT AND TOTAL ITEMS OUTLIERS (stricter bounds = less flagged transactions)
            total_time_spent = sum(time for _, time, _ in transaction)
            total_items = 0
            for department, _, _ in transaction:
                total_items += 1
            X = np.array([[total_time_spent, total_items]])
            distances = pairwise_distances_argmin_min(X, time_cluster_centroids)[1]

            if min(distances) > time_threshold:
                fraud_score += 2.0
                total_time_score += 2.0

            # CHECK TOTAL PRICE SPENT AND TOTAL ITEMS OUTLIERS (stricter bounds = less flagged transactions)
            total_price_spent = sum(price for _, _, price in transaction)
            total_items = 0
            for department, _, _ in transaction:
                total_items += 1
            X = np.array([[total_price_spent, total_items]])
            distances = pairwise_distances_argmin_min(X, price_cluster_centroids)[1]

            if min(distances) > price_threshold:
                fraud_score += 2.0
                total_price_score += 2.0

            # SCORE THRESHOLD
            if fraud_score >= score_threshold:
                flagged_transactions.append((idx, transaction))

        plot_rules = data_miner.mine_association_rules(support=0.1, lift=1)
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_rules["support"], plot_rules["lift"], color="blue", alpha=0.5)
        plt.xlabel("Support")
        plt.ylabel("Lift")
        plt.title("Association Rules: Support vs. Lift")
        plt.grid(True)
        plt.tight_layout()

        time_values = [
            time for transaction in test_transactions for _, time, _ in transaction
        ]
        plt.figure(figsize=(10, 6))
        sns.histplot(time_values, kde=True, color="blue")
        plt.xlabel("Transaction Time")
        plt.ylabel("Frequency")
        plt.title("Time Outliers Analysis")
        plt.tight_layout()

        price_values = [
            price for transaction in test_transactions for _, _, price in transaction
        ]
        plt.figure(figsize=(10, 6))
        sns.histplot(price_values, kde=True, color="red")
        plt.xlabel("Transaction Price")
        plt.ylabel("Frequency")
        plt.title("Price Outliers Analysis")
        plt.tight_layout()

        return (
            flagged_transactions,
            total_ar_score,
            total_seq_score,
            total_time_score,
            total_price_score,
        )

    def print_flagged_transactions(
        self,
        flagged_transactions,
        total_ar_score,
        total_seq_score,
        total_time_score,
        total_price_score,
    ):
        if len(flagged_transactions) == 0:
            print("No flagged transactions")
        elif len(flagged_transactions) <= 150:
            total_score = (
                total_ar_score + total_seq_score + total_time_score + total_price_score
            )
            print(
                f"Total Association Rules Score: {total_ar_score} ({total_ar_score / total_score * 100:.2f}%)"
            )
            print(
                f"Total Sequential Rules Score: {total_seq_score} ({total_seq_score / total_score * 100:.2f}%)"
            )
            print(
                f"Total Time Outliers Score: {total_time_score} ({total_time_score / total_score * 100:.2f}%)"
            )
            print(
                f"Total Price Outliers Score: {total_price_score} ({total_price_score / total_score * 100:.2f}%)"
            )
            print(f"{len(flagged_transactions)} Flagged Transactions")
            for idx, _ in flagged_transactions:
                transaction_number = idx.split()[1]
                print(f"{transaction_number}")
        else:
            random_selected_transactions = random.sample(flagged_transactions, 150)
            total_score = (
                total_ar_score + total_seq_score + total_time_score + total_price_score
            )
            print(
                f"Total Association Rules Score: {total_ar_score} ({total_ar_score / total_score * 100:.2f}%)"
            )
            print(
                f"Total Sequential Rules Score: {total_seq_score} ({total_seq_score / total_score * 100:.2f}%)"
            )
            print(
                f"Total Time Outliers Score: {total_time_score} ({total_time_score / total_score * 100:.2f}%)"
            )
            print(
                f"Total Price Outliers Score: {total_price_score} ({total_price_score / total_score * 100:.2f}%)"
            )
            print(
                f"{len(random_selected_transactions)} / {len(flagged_transactions)} Flagged Transactions"
            )
            for idx, _ in random_selected_transactions:
                transaction_number = idx.split()[1]
                print(f"{transaction_number}")


if __name__ == "__main__":
    data_miner = DataMiner("case I/supermarket.csv")
    fraud_detector = FraudDetector(data_miner)

    test_transactions, transaction_ids = fraud_detector.load_transactions(
        "case I/case48.csv"
    )
    (
        flagged_transactions,
        total_ar_score,
        total_seq_score,
        total_time_score,
        total_price_score,
    ) = fraud_detector.detect_fraud(
        test_transactions, transaction_ids, score_threshold=3.0
    )
    fraud_detector.print_flagged_transactions(
        flagged_transactions,
        total_ar_score,
        total_seq_score,
        total_time_score,
        total_price_score,
    )
