from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
import pendulum

default_args = {
    "owner": "bukovskiy7777",
    "start_date": pendulum.today('UTC').add(days=-2),  # запуск день назад
    "retries": 5,  # запуск таска до 5 раз, если ошибка
    #"retry_delay": datetime.timedelta(minutes=5),  # дельта запуска при повторе 5 минут
    "task_concurency": 1  # одновременно только 1 таск
}

piplines = {'train': {"schedule": "1 * * * *"},  # At minute 1 every hour
            "predict": {"schedule": "3 * * * *"}}  # At minute 3 every hour

def init_dag(dag, task_id):
    with dag:
        t1 = BashOperator(
            task_id=f"{task_id}",
            bash_command=f'python /home/oleksandr/apps/airflow-local/{task_id}.py')
    return dag

for task_id, params in piplines.items():
    # DAG - ациклический граф
    dag = DAG(task_id,
              schedule=params['schedule'],
              max_active_runs=1,
              default_args=default_args
              )
    init_dag(dag, task_id)
    globals()[task_id] = dag
