from prefect import task
from confluent_kafka import Consumer, KafkaError
import json
from datetime import datetime
import pandas as pd


class AIOpsPipeline:
    def __init__(self):
        self.data_buffer = []
        self.consumer = Consumer({
            'bootstrap.servers': 'localhost:29092',
            'group.id': 'ml-prediction-engine',
            'auto.offset.reset': 'latest'
        })

    @task(log_prints=True, retries=3)
    def ingest_data(self, topic='deployment-metrics'):
        self.consumer.subscribe([topic])
        msg = self.consumer.poll(1.0)

        if msg is None:
            return None
        if msg.error():
            raise KafkaError(msg.error())

        data = json.loads(msg.value().decode('utf-8'))
        validated_data = {
            'timestamp': datetime.now(),
            'source': topic,
            'data': data['data'],
            'quality_score': self._assess_quality(data['data'])
        }

        self.data_buffer.append(validated_data)
        return validated_data

    def _assess_quality(self, data):
        required_fields = [
            'deployment_id', 'timestamp', 'lines_changed', 'total_lines',
            'code_churn_ratio', 'author_success_rate', 'service_failure_rate_7d',
            'is_hotfix', 'touches_critical_path', 'test_coverage', 'build_duration_sec'
        ]

        completeness = sum(1 for field in required_fields if field in data) / len(required_fields)

        quality_checks = {
            'completeness': completeness,
            'validity': 1.0 if all(isinstance(data.get(field), (int, float, bool, str))
                                   for field in required_fields if field in data) else 0.8,
            'timeliness': 1.0 if datetime.now().hour < 20 else 0.8
        }

        return sum(quality_checks.values()) / len(quality_checks)
