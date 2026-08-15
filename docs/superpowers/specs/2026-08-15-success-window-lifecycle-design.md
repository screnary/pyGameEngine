# Success Window Lifecycle

## Scope

After the Agent reaches the target, keep the Pygame window open until the user
closes it. Preserve the current Behavior Tree, Environment dynamics, rendering,
and Milestone 2 metric definitions.

## Behavior

- Reaching the target finishes and saves the current Episode exactly once as
  `SUCCESS` with reason `target_reached`.
- The completed scene then freezes: the controller, environment, and recorder
  receive no further updates.
- Closing the window after success exits without writing a second
  `INTERRUPTED` result.
- Pressing `R` after success resets the Environment and controller and starts a
  fresh Episode without finishing the already-completed Episode again.
- Pressing `R` during an active Episode retains the existing
  `INTERRUPTED/manual_reset` behavior.
- Closing the window during an active Episode retains the existing
  `INTERRUPTED/window_closed` behavior.
- Timeout retains the current automatic-exit behavior.

## Implementation

`main.py` will use one `episode_finished` boolean. It becomes true after a
successful termination, gates simulation and recording updates, and returns to
false when `R` starts a new Episode. Event handling checks the flag before
finishing an Episode.

No new state-machine class, configuration option, dependency, or module is
needed.

## Targeted Verification

A small lifecycle test will run `main.main()` with Pygame events and a temporary
recorder output directory. It will verify that success keeps the loop alive,
window close does not add an interrupted result, and reset after success starts
a fresh Episode. Existing timeout and active-Episode interruption behavior will
remain intact.
