from confluent_kafka import Consumer, KafkaError
import json
from datetime import datetime

class AIOpsPipeline:
    def __init__(self):
        self.data_buffer = []
        self.consumer = Consumer({
            'bootstrap.servers': 'localhost:29092',
            'group.id': 'ml-prediction-engine',
            'auto.offset.reset': 'earliest',          # Changed from 'latest'
            'enable.auto.commit': True                # Added for auto offset commit
        })

    def ingest_data(self, topic='deployment-metrics', timeout=5, max_records=100):
        """
        Poll Kafka for up to `timeout` seconds or `max_records` messages.
        Returns a list of messages or None if nothing was received.
        """
        self.consumer.subscribe([topic])
        records = []
        elapsed = 0
        interval = 1  # poll interval in seconds

        while elapsed < timeout and len(records) < max_records:
            msg = self.consumer.poll(interval)
            if msg is None:
                elapsed += interval
                continue
            if msg.error():
                raise KafkaError(msg.error())

            try:
                data = json.loads(msg.value().decode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to decode message: {e}")
                continue

            validated_data = {
                'timestamp': datetime.now(),
                'source': topic,
                'data': data,
                'quality_score': self._assess_quality(data)
            }

            self.data_buffer.append(validated_data)
            records.append(validated_data)

        return records if records else None

    def _assess_quality(self, data):
        required_fields = [
            'deployment_id', 'timestamp', 'lines_changed', 'total_lines',
            'code_churn_ratio', 'author_success_rate', 'service_failure_rate_7d',
            'is_hotfix', 'touches_critical_path', 'test_coverage', 'build_duration_sec'
        ]
        completeness = sum(1 for field in required_fields if field in data) / len(required_fields)
        quality_checks = {
            'completeness': completeness,
            'validity': 1.0 if all(isinstance(data.get(f), (int, float, bool, str)) for f in required_fields if f in data) else 0.8,
            'timeliness': 1.0 if datetime.now().hour < 20 else 0.8
        }
        return sum(quality_checks.values()) / len(quality_checks)

    def close(self):
        """Gracefully close the Kafka consumer."""
        self.consumer.close()
