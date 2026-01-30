import json
import os
import glob

TRAJ_ROOT = "results/pareto_curves_405b"
DOMAINS = ["Transformer_Architectures", "Policy_Gradient_Methods", "Algorithmic_Fairness"]

def rollback_trajectories():
    for domain in DOMAINS:
        domain_path = os.path.join(TRAJ_ROOT, domain)
        if not os.path.exists(domain_path): continue
        
        # 找到所有轨迹文件
        traj_files = glob.glob(os.path.join(domain_path, "*.trajectory.json"))
        
        for fpath in traj_files:
            with open(fpath, "r") as f:
                traj = json.load(f)
            
            # 查找断点位置：我们要保留 Turn 0 到 Turn 14
            # 以及对应的 init 记录。
            # 简单起见，我们保留 Index 0 到 15 (因为 Turn 14 之前通常有 16 条记录)
            # 或者寻找第一个 Turn >= 15 的记录并截断
            
            new_traj = []
            for snapshot in traj:
                turn = snapshot["meta"]["turn"]
                if isinstance(turn, int) and turn >= 15:
                    break
                new_traj.append(snapshot)
            
            if len(new_traj) < len(traj):
                print(f"Rolling back {fpath}: {len(traj)} -> {len(new_traj)} snapshots")
                # 标记最后一个为未完成
                if new_traj:
                    new_traj[-1]["meta"]["is_completed"] = False
                
                with open(fpath, "w") as f:
                    json.dump(new_traj, f, indent=2)
                
                # 同时删除对应的 .raw.json 和 .emb.pkl，因为它们已经脏了
                prefix = fpath.replace(".trajectory.json", "")
                for extra in [".raw.json", ".emb.pkl"]:
                    if os.path.exists(prefix + extra):
                        os.remove(prefix + extra)
                        print(f"  Deleted {prefix + extra}")

if __name__ == "__main__":
    rollback_trajectories()
