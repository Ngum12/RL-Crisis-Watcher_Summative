#!/usr/bin/env python3
"""
Demo Launcher - Easy Access to All Crisis Response Demos
Choose which demonstration you want to run!
"""

import subprocess
import os

def show_menu():
    """Display demo selection menu"""
    print("🌍 AFRICA CRISIS RESPONSE AI - DEMO LAUNCHER")
    print("=" * 60)
    print()
    print("Choose your demonstration:")
    print()
    print("1. 🚁 Mobile Crisis Agent (RECOMMENDED FOR VIDEO)")
    print("   • Moving agent with pathfinding")
    print("   • Real geography and road networks")
    print("   • Crisis discovery and response coordination")
    print()
    print("2. 🧠 AI Decision Making Demo")
    print("   • Static analysis with live reasoning")
    print("   • Three-country focus: Cameroon, DRC, Sudan") 
    print("   • Shows AI thinking process")
    print()
    print("3. 🧠 Real RL Training (STABLE BASELINES3)")
    print("   • Train actual neural network agents")
    print("   • Save models, logs, and results")
    print("   • Professional RL with DQN, PPO, A2C")
    print()
    print("4. 📊 Simulated Training Demo")
    print("   • Quick training simulation")
    print("   • Generate comparison plots and reports")
    print("   • View training progress (63+ seconds)")
    print()
    print("5. 🧪 Test Trained Models")
    print("   • Load and test real trained models")
    print("   • Interactive model demonstration")
    print()
    print("6. 🤖 Live Model Visualization")
    print("   • Watch trained neural networks in action")
    print("   • Real-time decision making with visual feedback")
    print("   • Perfect for recording AI in action!")
    print()
    print("7. 🚁 Live Mobile Agent Control")
    print("   • Neural networks controlling moving agent")
    print("   • Real road networks, crisis detection, route planning")
    print("   • Watch AI navigate Africa and respond to crises!")
    print()
    print("8. 🎬 Video Recording Guide")
    print("   • Instructions for recording your assignment video")
    print("   • Tips for getting the best results")
    print()
    print("9. 📁 View Generated Files")
    print("   • Open results folder")
    print("   • View training plots and reports")
    print()
    print("0. Exit")
    print()

def main():
    """Main launcher function"""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-9): ").strip()
            print()
            
            if choice == "0":
                print("👋 Goodbye! Good luck with your assignment!")
                break
                
            elif choice == "1":
                print("🚁 Starting Mobile Crisis Agent...")
                print("🎯 Perfect for recording your assignment video!")
                print("Controls: SPACE=pause, R=reset, ESC=exit")
                print()
                input("Press ENTER to launch...")
                subprocess.run(["python", "mobile_crisis_agent.py"])
                
            elif choice == "2":
                print("🧠 Starting AI Decision Making Demo...")
                print("🎯 Shows transparent AI reasoning process")
                print("Controls: SPACE=pause, R=reset, ESC=exit")
                print()
                input("Press ENTER to launch...")
                subprocess.run(["python", "africa_crisis_demo.py"])
                
            elif choice == "3":
                print("🧠 Starting Real RL Training...")
                print("⏱️  This trains actual neural networks (may take several minutes)")
                print("🤖 Algorithms: DQN, PPO, A2C with Stable Baselines3")
                print("💾 Saves models, logs, and results")
                print()
                proceed = input("Continue? (y/N): ").strip().lower()
                if proceed in ['y', 'yes']:
                    subprocess.run(["python", "real_rl_training.py"])
                else:
                    print("Training cancelled.")
                    
            elif choice == "4":
                print("📊 Starting Simulated Training Demo...")
                print("⏱️  This will take 63+ seconds to complete")
                print("🤖 Training: DQN, PPO, REINFORCE, A2C")
                print()
                proceed = input("Continue? (y/N): ").strip().lower()
                if proceed in ['y', 'yes']:
                    subprocess.run(["python", "train_africa_models.py"])
                else:
                    print("Training cancelled.")
                    
            elif choice == "5":
                print("🧪 Testing Trained Models...")
                print("📦 Load and evaluate real trained models")
                print()
                input("Press ENTER to launch...")
                subprocess.run(["python", "test_trained_models.py"])
                    
            elif choice == "6":
                print("🤖 Starting Live Model Visualization...")
                print("📺 Watch your trained neural networks make real-time decisions!")
                print("🎬 Perfect for recording AI in action!")
                print()
                input("Press ENTER to launch...")
                subprocess.run(["python", "visualize_trained_models.py"])
                
            elif choice == "7":
                print("🚁 Starting Live Mobile Agent Control...")
                print("🎯 Watch your neural network control a moving agent through Africa!")
                print("📍 Real road networks, crisis detection, intelligent route planning")
                print()
                input("Press ENTER to launch...")
                subprocess.run(["python", "live_mobile_agent_visualization.py"])
                
            elif choice == "8":
                print("🎬 Opening Video Recording Guide...")
                subprocess.run(["python", "record_mobile_agent.py"])
                
            elif choice == "9":
                print("📁 Opening results folder...")
                if os.path.exists("results"):
                    if os.name == 'nt':  # Windows
                        subprocess.run(["explorer", "results"])
                    else:  # Linux/Mac
                        subprocess.run(["xdg-open", "results"])
                    
                    print("📊 Generated files:")
                    if os.path.exists("results/africa_training_comparison.png"):
                        print("   ✓ Training comparison plots")
                    if os.path.exists("reports"):
                        print("   ✓ Detailed analysis reports")
                    print("   ✓ Raw training data (JSON)")
                else:
                    print("❌ No results found. Run training first (option 3).")
                    
            else:
                print("❌ Invalid choice. Please enter 0-9.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")
            
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()