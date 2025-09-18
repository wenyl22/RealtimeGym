import time
import os
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

class Logger:
    def __init__(self, args, log_dir, seed, real_seed):
        self.log_dir = log_dir
        self.seed = seed
        self.real_seed = real_seed
        self.game_log = []
        self.streaming_log = []
        self.streaming_entry = None
        self.game_start_time = time.time()
        
    def log_turn_start(self, env, meta_control):
        self.current_turn = env.env.game_turn
        self.turn_start_time = time.time()
        turn_entry = {
            "type": "turn_start",
            "turn": self.current_turn,
            "timestamp": self.turn_start_time,
            "relative_time": self.turn_start_time - self.game_start_time,
            "meta_control": meta_control,
            "game_state": env.env.state_string()
        }
        self.game_log.append(turn_entry)
        self.save_game_log()
        
    def log_action_execution(self, env, action):
        current_time = time.time()
        self.current_reward= env.env.reward
        action_entry = {
            "type": "action_execution",
            "turn": self.current_turn,
            "timestamp": current_time,
            "relative_time": current_time - self.game_start_time,
            "turn_relative_time": current_time - self.turn_start_time,
            "action": action,
            "reward": self.current_reward,
            "game_state": env.env.state_string()
        }
        self.game_log.append(action_entry)
        self.save_game_log()
        
    def log_streaming_content(self, agent_type, content, append):
        current_time = time.time()
        if self.streaming_entry is None or not append or len(self.streaming_entry['content']) > 30:
            if self.streaming_entry is not None:
                self.streaming_log.append(self.streaming_entry)
            self.streaming_entry = {
                "type": "streaming",
                "agent": agent_type,
                "turn": self.current_turn,
                "timestamp": current_time,
                "relative_time": current_time - self.game_start_time,
                "turn_relative_time": current_time - self.turn_start_time,
                "content": "",
                "append": append
            }
        self.streaming_entry['content'] += content
        if len(self.streaming_log) % 100 == 0:
            self.save_streaming_log()
            
    def save_game_log(self):
        with open(f"{self.log_dir}/game_{self.seed}.json", "w") as f:
            json.dump(self.game_log, f, indent=2)
            
    def save_streaming_log(self):
        with open(f"{self.log_dir}/streaming_{self.seed}.json", "w") as f:
            json.dump(self.streaming_log, f, indent=2)
            
    def save_final_logs(self):
        self.save_game_log()
        self.save_streaming_log()
            
class Replay:
    def __init__(self, log_dir, seed):
        self.log_dir = log_dir
        self.seed = seed
        self.load_logs()
        
    def load_logs(self):
        with open(f"{self.log_dir}/game_{self.seed}.json", "r") as f:
            self.game_log = json.load(f)        
        with open(f"{self.log_dir}/streaming_{self.seed}.json", "r") as f:
            self.streaming_log = json.load(f)
    def get_state_at_time(self, time_point):
        relevant_entries = [entry for entry in self.game_log 
                          if entry['relative_time'] <= time_point]
        if not relevant_entries:
            return "Game not started"
        latest_entry = max(relevant_entries, key=lambda x: x['relative_time'])
        return latest_entry.get('game_state', 'No state available')
    def generate_demo_video(self, output_file="game_demo.mp4", fps=5, time_compression=10):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # 只使用一个subplot
        
        total_time = max([entry['relative_time'] for entry in self.game_log])
        compressed_duration = total_time / time_compression
        total_frames = int(compressed_duration * fps)
        
        def animate(frame):
            print(frame, total_frames)
            current_compressed_time = frame / fps
            current_real_time = current_compressed_time * time_compression
            ax.clear()
            
            # 显示游戏状态
            current_state = self.get_state_at_time(current_real_time)
            ax.text(0.5, 0.5, current_state, ha='center', va='center', 
                    fontsize=12, wrap=True, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            # 在右上角显示token信息
            token_text = self.get_token_text(current_real_time)
            ax.text(0.98, 0.98, token_text, ha='right', va='top', 
                    fontsize=10, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
            
            # 显示时间信息
            ax.text(0.02, 0.98, f"Time: {current_real_time:.2f}s", 
                    ha='left', va='top', fontsize=10, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            ax.set_title(f"Game State (Time: {current_real_time:.2f}s)")
            ax.axis('off')
            
        ani = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                    interval=1000//fps, repeat=False)
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Game Logger'))
        ani.save(f"{self.log_dir}/{output_file}", writer=writer)
        plt.close()
        print(f"Demo video saved as {self.log_dir}/{output_file}")
        print(f"Original duration: {total_time:.2f}s, Compressed to: {compressed_duration:.2f}s")

    def get_token_text(self, current_time):
        """获取当前时间点的token信息文本"""
        relevant_streaming = [entry for entry in self.streaming_log 
                            if entry['relative_time'] <= current_time]
        
        if not relevant_streaming:
            return "No token data"
        
        agent_tokens = {}
        for entry in relevant_streaming:
            agent = entry['agent']
            if agent not in agent_tokens:
                agent_tokens[agent] = ""
            
            if entry['append']:
                agent_tokens[agent] += entry['content']
            else:
                agent_tokens[agent] = entry['content']
        if not agent_tokens:
            return "No token data"        
        token_lines = ["Token Usage:"]
        for agent, text in agent_tokens.items():
            token_lines.append(f"{agent}: {int(0.3 * len(text))} tokens")
        
        return "\n".join(token_lines)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]
    else:
        log_dir = "walltime-logs/freeway_M_parallel_360_180_T"
    
    replay = Replay(log_dir=log_dir, seed = 0)
    replay.generate_demo_video(output_file="game_replay.mp4", fps=4, time_compression=200)