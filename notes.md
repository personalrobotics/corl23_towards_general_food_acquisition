time lost tracking
    iterate through the messages, subtract time between two messages, use the average to create a threshold
    n times the frequency we expect, if it's more, then we've some number of messages and that is time lost tracking
    decided on 0.1s to be the threshold, iterate through all the messages keeping a running sum and report that at the end of the bag

end time
    just get the last timestamp

start time
    pick time where fork is stationary-ish (within some delta), improves on continuous tracking 
        filters out fork on table points, use height coordinate
    pick first few seconds of continuous tracking
    pick when force is on forque and backtrack

    we made them hold it in the air for some time (0.5-1.0)
        no guarantee of no tracking when the fork is on the table
        how can we determine how the height was
            maybe it's not z, could be x or y

    continuous tracking the whole time (including the table)

    some threshold of minimal movement

    three parameters: specify them all the top so it's easy to see how to change them

        amount of time for conditions
        height off table
        minimum movement during that 
    
    [preprocess]
    figure out which direction is up in the poses (replay back in rviz)
    height threshold should be 4cm for now
    time threshold is 1sec
    movement threshold is 0.5-1cm

    [processing]
    get the height coord of the first table message
    for each timestamp we need the forque pose (x,y,z)
        check if the height is above the threshold
        if yes (hold this timestamp as start of continuous tracking, also hold x,y,z)        
        reset if we lose tracking
        once we have a period of time that is continuous tracking for the threshold
            now check the euclidean distance between this and the start continuous tracking (minimal motion parameter)
            if that's less than the threshold
                the end of this is the start time

    [optimizations for future?]
    - if a user goes up and down, doesn't account for that 
        we currently stop caring about height after first message above threshold
        ++ solutions: also reset continuous time tracking if we go below the height threshold
    - we don't know how long the user held the fork in the air, may have held for 5 seconds or 1 seconds    
        we only look at the first n seconds of tracking and stop after we hit that threshold
    - (once 1 is implemented) if we have perfect tracking, we hit the threshold on picking up, then time resets and then we'll never have a start time
        tracking works but we start as soon as we hit threshold and not when we stop moving
        +++ solve by storing everything (2 loops approach with breaks when conditions are not met)

[replace optimizations and processing with paper notes]
implement this then compare with camera data from the bag (both pose and image) for 10 bag files as spot check

