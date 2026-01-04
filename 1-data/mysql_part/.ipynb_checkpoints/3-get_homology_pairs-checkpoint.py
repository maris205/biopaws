import pymysql
import hashlib
from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML
import time
import random

# ==================== 数据库配置 ====================
HOST = ""
PORT = 3306
USER = ""
PASSWORD = ""
DATABASE = "ionet"
CHARSET = "utf8mb4"

# ==================== BLAST 参数 ====================
MAX_HOMOLOGS_PER_QUERY = 5
E_VALUE_THRESH = 1e-10
IDENTITY_THRESH = 30.0
HITLIST_SIZE = 20

# ==================== 数据库连接 ====================
connection = pymysql.connect(
    host=HOST,
    port=PORT,
    user=USER,
    password=PASSWORD,
    database=DATABASE,
    charset=CHARSET,
    autocommit=False,          # 手动提交
    cursorclass=pymysql.cursors.DictCursor  # 可选：返回字典，便于阅读
)

def compute_pair_md5(seq1, seq2):
    if seq1 < seq2:
        combined = seq1 + seq2
    else:
        combined = seq2 + seq1
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

try:
    with connection.cursor() as cursor:
        while True:
            # ==================== 关键修改：随机 + 锁机制 ====================
            try:
                # 开始事务 + 行锁：确保多实例不会同时选中同一行
                cursor.execute("START TRANSACTION")
                cursor.execute("""
                    SELECT id, sentence1 
                    FROM protein_seq 
                    WHERE is_process = 0 
                    ORDER BY RAND() 
                    LIMIT 1 
                    FOR UPDATE  -- 重要！锁住选中的行，直到事务结束
                """)
                row = cursor.fetchone()

                if row is None:
                    connection.commit()  # 提交空事务
                    print("所有序列已处理完成，没有剩余 is_process=0 的记录。")
                    break

                query_id_in_db = row['id']
                query_seq = row['sentence1']

                # 立即提交锁事务，释放行锁（但还没标记处理）
                connection.commit()

            except Exception as e:
                connection.rollback()
                print(f"获取随机序列时出错: {e}")
                time.sleep(5)
                continue
            # ==============================================================

            print(f"\n正在处理数据库 ID={query_id_in_db} 的蛋白序列 (长度={len(query_seq)})")

            try:
                print("  正在运行 NCBI BLAST...")
                result_handle = NCBIWWW.qblast(
                    program="blastp",
                    database="swissprot",
                    sequence=query_seq,
                    expect=E_VALUE_THRESH,
                    hitlist_size=HITLIST_SIZE
                )

                blast_records = NCBIXML.parse(result_handle)
                blast_record = next(blast_records, None)

                inserted_count = 0
                if blast_record:
                    for alignment in blast_record.alignments:
                        if inserted_count >= MAX_HOMOLOGS_PER_QUERY:
                            break
                        hit_id = alignment.hit_id

                        for hsp in alignment.hsps:
                            if inserted_count >= MAX_HOMOLOGS_PER_QUERY:
                                break

                            hit_seq_aligned = hsp.sbjct
                            identity = (hsp.identities / hsp.align_length) * 100

                            if identity < IDENTITY_THRESH:
                                continue

                            # 排除自身（粗略判断）
                            hit_acc = hit_id.split("|")[1] if '|' in hit_id else ""
                            if hit_acc and query_seq.startswith(hit_acc):
                                continue

                            pair_md5 = compute_pair_md5(query_seq, hit_seq_aligned)

                            insert_sql = """
                                INSERT IGNORE INTO protein_pair
                                (sentence1, sentence2, pair_md5, label)
                                VALUES (%s, %s, %s, 1)
                            """
                            cursor.execute(insert_sql, (query_seq, hit_seq_aligned, pair_md5))
                            if cursor.rowcount > 0:
                                inserted_count += 1
                                print(f"    → 插入正样本对 (label=1, identity={identity:.1f}%)")

                print(f"  本序列共插入 {inserted_count} 条正样本对。")

            except Exception as e:
                print(f"  BLAST 出错: {e}，仍将标记为已处理（避免重复尝试）")

            # ==================== 标记为已处理 ====================
            try:
                cursor.execute("""
                    UPDATE protein_seq 
                    SET is_process = 1 
                    WHERE id = %s
                """, (query_id_in_db,))
                connection.commit()
                print(f"  已将数据库 ID={query_id_in_db} 标记为已处理 (is_process=1)")
            except Exception as e:
                connection.rollback()
                print(f"  更新 is_process 失败: {e}")

            # NCBI 限速等待
            sleep_time = 3 + random.uniform(0, 3)
            print(f"  等待 {sleep_time:.1f} 秒避免 NCBI 限速...\n")
            time.sleep(sleep_time)

except Exception as e:
    print(f"程序异常: {e}")
    connection.rollback()
finally:
    connection.close()
    print("数据库连接已关闭，程序结束。")