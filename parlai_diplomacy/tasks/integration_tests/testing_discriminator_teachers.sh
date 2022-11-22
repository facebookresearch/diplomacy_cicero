# uses diplom dd to test integration of discrimanator teachers
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#
# This is not a real test (doesn't run on CircleCI, for example),
# It is just a script to check if things are in-place for
# running Discriminator training teachers
echo 'Testing discrimantor teachers.'
echo 'This test is successful if it does not crash and you see "successfully finished" message'
echo

TEACHER_MESSAGE_FORMATS=(
    "message_history"
    "message_history_state"
    "message_history_orderhistorysincelastmovementphase_shortstate"
)


echo "-------------------------------------------------"
echo "Testing valid teachers"
for teacher_type in corrupted real
do
    for formatting in ${TEACHER_MESSAGE_FORMATS[@]}
    do
        echo "--------------------"
        teacher="valid${teacher_type}_${formatting}_dialoguediscriminator_evaldump"
        echo "Testing ${teacher}"
        diplom dd -t ${teacher} -dt valid 1>/dev/null || exit 1
        echo "${teacher} finished successfully."
    done
done
echo "-------------------------------------------------"
echo "Finished Testing valid teachers"

TEACHER_DATA_TYPES=(
    ""
    "corruptedreceiver_"
    "corruptedentity_"
    "incorrectphase_"
    "incorrectgame_"
    "repeatmessage_"
)

echo
echo "Testing the chunk (training) teachers"
for teacher_type in ${TEACHER_DATA_TYPES[@]}
do
    for formatting in ${TEACHER_MESSAGE_FORMATS[@]}
    do
        echo "--------------------"
        teacher="${teacher_type}${formatting}_dialoguediscriminator_chunk"
        echo "Testing ${teacher}"
        diplom dd -t ${teacher} -dt train:stream 1>/dev/null || exit 1
        echo "${teacher} finished successfully."
    done
done

echo '================================================'
echo 'All tests successfully finished.'