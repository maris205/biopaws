import pymysql
import hashlib
from Bio import SeqIO
from Bio.Align import PairwiseAligner
import random
import time

# ==================== 数据库配置 ====================
HOST = ""
PORT = 3306
USER = ""
PASSWORD = ""
DATABASE = "ionet"
CHARSET = "utf8mb4"

# ==================== 负样本生成参数 ====================
MAX_NEG_PER_QUERY = 5           # 每条查询生成 5 个负样本
IDENTITY_THRESHOLD = 25.0       # identity < 25% 视为非同源
LENGTH_TOLERANCE = 0.3           # 长度差异不超过 30%
MIN_LENGTH = 50
MAX_TRIALS = 500                # 最多尝试 500 次随机采样

# 完整 Swiss-Prot 文件路径
FULL_SWISSPROT = "uniprot_sprot.fasta"

# ==================== 全局比对器 ====================
aligner = PairwiseAligner(mode='global')
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.gap_score = -5
aligner.extend_gap_score = -1

# ==================== 加载完整 Swiss-Prot ====================
print("Loading full Swiss-Prot database... (this may take a minute)")
full_records = [r for r in SeqIO.parse(FULL_SWISSPROT, "fasta") if len(r.seq) >= MIN_LENGTH]
print(f"Loaded {len(full_records)} sequences from Swiss-Prot.")

# ==================== 数据库连接 ====================
connection = pymysql.connect(
    host=HOST,
    port=PORT,
    user=USER,
    password=PASSWORD,
    database=DATABASE,
    charset=CHARSET,
    autocommit=False
)

def compute_pair_md5(seq1, seq2):
    """有序拼接计算 MD5，确保去重一致"""
    if seq1 < seq2:
        combined = seq1 + seq2
    else:
        combined = seq2 + seq1
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

try:
    with connection.cursor() as cursor:
        while True:
            # 1. 取一条 is_dis_process = 0 的序列（用于负样本处理）
            cursor.execute("""
                SELECT id, sentence1 
                FROM protein_seq 
                WHERE is_dis_process = 0 
                ORDER BY id 
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row is None:
                print("所有序列的负样本处理已完成，没有剩余 is_dis_process=0 的记录。")
                break

            query_id_in_db, query_seq = row
            q_len = len(query_seq)
            print(f"\n正在处理负样本 - 数据库 ID={query_id_in_db} (长度={q_len})")

            found = 0
            trials = 0

            while found < MAX_NEG_PER_QUERY and trials < MAX_TRIALS:
                trials += 1
                cand_rec = random.choice(full_records)

                c_seq = str(cand_rec.seq)
                if c_seq == query_seq:  # 避免完全相同（极少发生）
                    continue

                c_len = len(c_seq)

                # 长度过滤
                if abs(q_len - c_len) / ((q_len + c_len) / 2) > LENGTH_TOLERANCE:
                    continue

                # 全局比对
                try:
                    alignments = aligner.align(query_seq, c_seq)
                    if not alignments:
                        continue
                    alignment = alignments[0]

                    identical = sum(a == b and a != '-' and b != '-' 
                                    for a, b in zip(str(alignment[0]), str(alignment[1])))
                    aligned_len = (len(str(alignment[0])) - 
                                   str(alignment[0]).count('-') - 
                                   str(alignment[1]).count('-'))
                    identity = 100.0 * identical / aligned_len if aligned_len > 0 else 0

                    if identity < IDENTITY_THRESHOLD:
                        pair_md5 = compute_pair_md5(query_seq, c_seq)

                        # 插入负样本对，label=0
                        insert_sql = """
                            INSERT IGNORE INTO protein_pair 
                            (sentence1, sentence2, pair_md5, label) 
                            VALUES (%s, %s, %s, 0)
                        """
                        cursor.execute(insert_sql, (query_seq, c_seq, pair_md5))
                        if cursor.rowcount > 0:
                            found += 1
                            print(f"    → 插入负样本对 (label=0, identity={identity:.1f}%)")

                except Exception as e:
                    print(f"    比对异常: {e}")
                    continue

            print(f"  本序列共插入 {found} 条负样本对（尝试 {trials} 次）")

            # 2. 标记为已处理负样本（使用新字段）
            cursor.execute("""
                UPDATE protein_seq 
                SET is_dis_process = 1 
                WHERE id = %s
            """, (query_id_in_db,))

            connection.commit()
            print(f"  已将数据库 ID={query_id_in_db} 的负样本处理标记为完成 (is_dis_process=1)\n")

            # 短暂休眠，避免 CPU 过载
            time.sleep(0.1)

except Exception as e:
    print(f"数据库操作异常: {e}")
    connection.rollback()
finally:
    connection.close()
    print("数据库连接已关闭，负样本生成程序结束。")