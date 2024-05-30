import subprocess


def run_multiple():
    dtu_objects = [
        '110_rabbit', '37_scissors', '63_fruits', '83_smurf', '97_cans',
        '106_birds', '69_snowman', '40_block', '65_skull', '105_plush',
        '114_buddha', '55_bunny', '24_redhouse', '122_owl', '118_angel']
    idx = 0
    upsamples = [2000, 2500, 3000]
    rebuilds = [500, 1000, 1500]
    while idx < len(dtu_objects):
        run_name = dtu_objects[idx]
        try:
            subprocess.run(
                ["dvc", "exp", "run", "-f",
                 "--downstream", "run-eval",
                 "--name", f"{run_name}",
                 "-S", f"run.run_name={run_name}",
                 "-S", f"run.upsample_iterations={upsamples}",
                 "-S", f"run.rebuild_iterations={rebuilds}",
                 "-S", "run.iterations=3500"],
                timeout=1000)
        except subprocess.TimeoutExpired:
            print(f"Had to stop exp {run_name}, rerunning")
            continue
        idx += 1


if __name__ == "__main__":
    run_multiple()
