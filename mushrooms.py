import tensorflow as tf
import numpy as np
import random

def main():
    label_index ={
        'p': 0, #Poisonous
        'e': 1  #Edible
    }
    cap_shape_index={
        'b': 0, #Bell
        'c': 1, #Conical
        'x': 2, #convex
        'f': 3, #flat
        'k': 4, #knobbed
        's': 5  #sunken
    }
    cap_surface_index ={
        'f': 0, #fibrous
        'g': 1, #grooves
        'y': 2, #scaly
        's': 3  #smooth
    }
    cap_color_index={
        'n': 0, #brown
        'b': 1, #buff
        'c': 2, #cinnamon
        'g': 3, #gray
        'r': 4, #green
        'p': 5, #pink
        'u': 6, #purple
        'e': 7, #red
        'w': 8, #white
        'y': 9  #yellow
    }
    bruises_index={
        't': 0, #Yes bruises
        'f': 1  #No bruises
    }
    odor_index={
        'a': 0, #
        'l': 1, #
        'c': 2, #
        'y': 3, #
        'f': 4, #
        'm': 5, #
        'n': 6, #
        'p': 7, #
        's': 8  #
    }
    gill_attachment_index={
        'a': 0, #
        'd': 1, #
        'f': 2, #
        'n': 3  #
    }
    gill_spacing_index={
        'c': 0, #
        'w': 1, #
        'd': 2  #
    }
    gill_size_index={
        'b': 0, #broad
        'n': 1  #narrow
    }
    gill_color_index={
        'k': 0, #
        'n': 1, #
        'b': 2, #
        'h': 3, #
        'g': 4, #
        'r': 5, #
        'o': 6, #
        'p': 7, #
        'u': 8, #
        'e': 9, #
        'w': 10,#
        'y': 11 #
    }
    stalk_shape_index={
        'e': 0, #
        't': 1  #
    }
    stalk_root_index={
        'b': 0, #
        'c': 1, #
        'u': 2, #
        'e': 3, #
        'z': 4, #
        'r': 5, #
        '?': 6  #
    }
    stalk_surf_above_ring_index={
        'f': 0, #
        'y': 1, #
        'k': 2, #
        's': 3  #
    }
    stalk_surf_below_ring_index={
        'f': 0, #
        'y': 1, #
        'k': 2, #
        's': 3  #
    }
    stalk_color_above_ring_index={
        'n': 0, #
        'b': 1, #
        'c': 2, #
        'g': 3, #
        'o': 4, #
        'p': 5, #
        'e': 6, #
        'w': 7, #
        'y': 8  #
    }
    stalk_color_below_ring_index={
        'n': 0, #
        'b': 1, #
        'c': 2, #
        'g': 3, #
        'o': 4, #
        'p': 5, #
        'e': 6, #
        'w': 7, #
        'y': 8  #
    }
    veil_type_index={
        'p': 0, #
        'u': 1  #
    }
    veil_color_index={
        'n': 0, #
        'o': 1, #
        'w': 2, #
        'y': 3  #
    }
    ring_number_index={
        'n': 0, #
        'o': 1, #
        't': 2  #
    }
    ring_type_index={
        'c': 0, #
        'e': 1, #
        'f': 2, #
        'l': 3, #
        'n': 4, #
        'p': 5, #
        's': 6, #
        'z': 7  #
    }
    spore_print_color_index={
        'k': 0, #
        'n': 1, #
        'b': 2, #
        'h': 3, #
        'r': 4, #
        'o': 5, #
        'u': 6, #
        'w': 7, #
        'y': 8  #
    }
    population_index={
        'a': 0, #
        'c': 1, #
        'n': 2, #
        's': 3, #
        'v': 4, #
        'y': 5  #
    }   
    habitat_index={
        'g': 0, #
        'l': 1, #
        'm': 2, #
        'p': 3, #
        'u': 4, #
        'w': 5, #
        'd': 6  #
    }   

    print ("Is this mushroom edible? Let us train...")

    data = []
    labels = []

    with open("agaricus-lepiota.data","r") as input_file:
        for line in input_file:
            if(len(line.strip()) == 0):
                continue

            full_data_line = line.strip().split(",")

            data_line = full_data_line[1:23]
            label_line = full_data_line[0]

            # print ("Data Line: ", data_line)
            # print ("Label Line: ", label_line)
            # break
            if label_line in label_index:
                label_line = label_index[label_line]
            else:
                print("Bad label line, bad!!!", line)
            if data_line[0] in cap_shape_index:
                data_line[0] = cap_shape_index[data_line[0]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[1] in cap_surface_index:
                data_line[1] = cap_surface_index[data_line[1]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[2] in cap_color_index:
                data_line[2] = cap_color_index[data_line[2]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[3] in bruises_index:
                data_line[3] = bruises_index[data_line[3]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[4] in odor_index:
                data_line[4] = odor_index[data_line[4]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[5] in gill_attachment_index:
                data_line[5] = gill_attachment_index[data_line[5]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[6] in gill_spacing_index:
                data_line[6] = gill_spacing_index[data_line[6]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[7] in gill_size_index:
                data_line[7] = gill_size_index[data_line[7]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[8] in gill_color_index:
                data_line[8] = gill_color_index[data_line[8]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[9] in stalk_shape_index:
                data_line[9] = stalk_shape_index[data_line[9]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[10] in stalk_root_index:
                data_line[10] = stalk_root_index[data_line[10]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[11] in stalk_surf_above_ring_index:
                data_line[11] = stalk_surf_above_ring_index[data_line[11]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[12] in stalk_surf_below_ring_index:
                data_line[12] = stalk_surf_below_ring_index[data_line[12]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[13] in stalk_color_above_ring_index:
                data_line[13] = stalk_color_above_ring_index[data_line[13]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[14] in stalk_color_below_ring_index:
                data_line[14] = stalk_color_below_ring_index[data_line[14]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[15] in veil_type_index:
                data_line[15] = veil_type_index[data_line[15]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[16] in veil_color_index:
                data_line[16] = veil_color_index[data_line[16]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[17] in ring_number_index:
                data_line[17] = ring_number_index[data_line[17]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[18] in ring_type_index:
                data_line[18] = ring_type_index[data_line[18]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[19] in spore_print_color_index:
                data_line[19] = spore_print_color_index[data_line[19]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[20] in population_index:
                data_line[20] = population_index[data_line[20]]
            else:
                print("Bad data line, bad!!!", line)
            if data_line[21] in habitat_index:
                data_line[21] = habitat_index[data_line[21]]
            else:
                print("Bad data line, bad!!!", line)

            data.append(data_line)
            labels.append(label_line)

    print("data", len(data))
    print("labels", len(labels))

    dataset = list(zip(data, labels))
    random.shuffle(dataset)
    test_length = int(len(dataset) * 0.67)

    print("test_length", test_length)
    train_dataset = dataset[:test_length]
    test_dataset = dataset[test_length:]

    x_size = 22
    out_size = 2
    num_nodes = 200

    # inputs needs to be type float for matmul to work...
    inputs = tf.placeholder("float", shape=[None, x_size])
    labels = tf.placeholder("int32", shape=[None])

    weights1 = tf.get_variable("weight1", shape=[x_size, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable("bias1", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))
    
    layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)

    weights2 = tf.get_variable("weight2", shape=[num_nodes, num_nodes], initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable("bias2", shape=[num_nodes], initializer=tf.constant_initializer(value=0.0))
    
    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)

    weights3 = tf.get_variable("weight3", shape=[num_nodes, out_size], initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable("bias3", shape=[out_size], initializer=tf.constant_initializer(value=0.0))
    
    outputs = tf.matmul(layer2, weights3) + bias3

    # Back prop

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(labels, out_size), logits=outputs))
    train = tf.train.AdamOptimizer().minimize(loss)

    predictions = tf.argmax(tf.nn.softmax(outputs), axis=1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(5000):
            batch = random.sample(train_dataset, 25)
            inputs_batch, labels_batch = zip(*batch)
            loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: inputs_batch, labels: labels_batch})

            # print("Prediction output", prediction_output)
            # print("Labels batch", labels_batch)

            # accuracy = np.mean(labels_batch == prediction_output)

            # print("train", "loss", loss_output, "accuracy", accuracy)

        # test our trained model with test data
        batch = random.sample(test_dataset, 100)
        inputs_batch, labels_batch = zip(*batch)
        loss_output, prediction_output, _ = sess.run([loss, predictions, train], feed_dict={inputs: inputs_batch, labels: labels_batch})
        accuracy = np.mean(labels_batch == prediction_output)

        f = open('output.txt', 'w')
        print("Prediction output: ", prediction_output)
        myString = 'Prediction output: ' + np.array_str(prediction_output)
        f.write(myString)
        f.write('\n\n')
        print("Labels batch: ", labels_batch)
        t = ' '.join(str(v) for v in labels_batch)
        myString = 'Labels batch:      [' + t[0:73] + '\n ' + t[74:147] + '\n ' + t[148:] + ']\n\n'
        f.write(myString)
        print("Loss: ", loss_output)
        myString = 'Loss: ' + np.array_str(loss_output)
        f.write(myString + '\n\n')
        print("Accuracy: ", accuracy)
        myString = 'Accuracy: ' + np.array_str(accuracy)
        f.write(myString)
        # I believe this is also training, how to run without training?
        f.close

if __name__ == "__main__":
    main()
