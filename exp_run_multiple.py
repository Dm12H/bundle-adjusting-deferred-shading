import subprocess


def run_multiple():
    dtu_objects = [
        '110_rabbit', '37_scissors', '63_fruits', '83_smurf', '97_cans',
        '106_birds', '69_snowman', '40_block', '65_skull', '105_plush',
        '114_buddha', '55_bunny', '24_redhouse', '122_owl', '118_angel']
    idx = 0
    while idx < len(dtu_objects):
        cur_obj = dtu_objects[idx]
        try:
            subprocess.run(
                ["dvc", "exp", "run",
                 "--name", cur_obj,
                 "-S", f"run.run_name={cur_obj}"],
                timeout=1000)
        except subprocess.TimeoutExpired:
            print(f"Had to stop exp {cur_obj}, rerunning")
        idx += 1


if __name__ == "__main__":
    run_multiple()