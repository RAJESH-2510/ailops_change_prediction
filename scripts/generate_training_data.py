import pandas as pd
import random
from datetime import datetime, timedelta

def generate_dataset_record():
    deployment_id = f"build-{random.randint(1000, 9999)}"
    timestamp = (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
    lines_changed = random.randint(100, 2000)
    total_lines = random.randint(5000, 20000)
    code_churn_ratio = round(lines_changed / total_lines, 3)
    author_success_rate = round(random.uniform(0.7, 0.95), 2)
    service_failure_rate_7d = round(random.uniform(0.01, 0.1), 3)
    is_hotfix = random.choice([True, False])
    touches_critical_path = random.choice([True, False])
    test_coverage = random.randint(60, 90)
    build_duration_sec = random.randint(120, 300)
    risk_score = round(random.uniform(0.1, 0.9), 2)
    failure_label = 1 if risk_score > 0.7 and random.random() > 0.3 else 0

    return {
        'deployment_id': deployment_id,
        'timestamp': timestamp,
        'lines_changed': lines_changed,
        'total_lines': total_lines,
        'code_churn_ratio': code_churn_ratio,
        'author_success_rate': author_success_rate,
        'service_failure_rate_7d': service_failure_rate_7d,
        'is_hotfix': is_hotfix,
        'touches_critical_path': touches_critical_path,
        'test_coverage': test_coverage,
        'build_duration_sec': build_duration_sec,
        'risk_score': risk_score,
        'failure_label': failure_label
    }

def generate_training_data(num_records=1000):
    data = []
    for _ in range(num_records):
        record = generate_dataset_record()
        data.append(record)

    df = pd.DataFrame(data)
    df.to_csv('data/training_data.csv', index=False)
    print(f"Generated {num_records} records to data/training_data.csv")

if __name__ == "__main__":
    generate_training_data(1000)
