import pickle
import psycopg2
from multiprocessing import Lock


class SqlLogger:
    user = 'yongbin'
    dbname = 'mmocr'
    host = 'localhost'
    password = ''
    iteration = 0
    global_var = dict()
    flag = False

    @staticmethod
    def log_tensor(attn_map, img, target, output_step, step):
        pickled_attn_map = pickle.dumps(attn_map)
        pickled_img = pickle.dumps(img)

        conn = psycopg2.connect(
            dbname=SqlLogger.dbname,
            user=SqlLogger.user,
            password=SqlLogger.password,
            host=SqlLogger.host
        )

        cur = conn.cursor()
        cur.execute(
            '''INSERT INTO attn_vis (attn_map, img, target, output_step, iteration, step)
            VALUES (%(attn_map)s, %(img)s, %(target)s, %(output_step)s, %(iteration)s, %(step)s);
            ''',
            dict(
                attn_map=pickled_attn_map,
                img=pickled_img,
                target=target,
                output_step=output_step,
                iteration=SqlLogger.iteration,
                step=step
            )
        )
        conn.commit()
