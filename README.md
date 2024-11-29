# Bearing Fault Detection Based on CNN, LSTM, GRU, and Self-Attention

## 1 Data Collection and Augmentation of Equipment

In the rotating parts of motion components such as machining centers and robotic arms, bearings play a crucial role. The health condition of the bearings directly affects the operational efficiency and precision of the machining centers and robotic arms, hence the importance of fault detection for bearings. Common types of bearing failures include inner race faults, outer race faults, and ball faults. These failures typically occur due to the local damaged parts of the balls hitting the outer or inner races, or the damaged balls hitting the inner and outer races, thereby exciting high-frequency resonance between the bearings and the responding sensors. By using accelerometers to collect the vibration signals of the bearings and employing machine learning and deep learning methods to identify and judge the collected data, these faults can be effectively diagnosed.

​     ![image-20241129101151134](picture\image-20241129101151134.png)

**Figure 1-1 Bearing Structure Schematic**

In addition, monitoring parameters such as the operating temperature of the robotic arm, motor drive voltage and current, and positional posture are also important means of judging its operating state. By collecting these data and analyzing their range and development trends, potential problems can be identified in time, so that corresponding maintenance measures can be taken to ensure the stable operation of the machining center.

This paper uses the open-source bearing fault diagnosis dataset from Purdue University [57], which includes vibration signal data of nine different fault types. Each fault type covers three dimensions of measurement data: drive end (DE_time), fan end (FE_time), and base (BE_time), providing a comprehensive data analysis foundation for the study.

![image-20241129095324236](picture\image-20241129095324236.png)

**Figure 1-2 Purdue University Bearing Fault Detection Test Bench [57]**

Specifically, each fault type in the dataset includes about 85,000 data points, as shown in Figure 1-3, where the data is represented in red, green, and blue for DE_time, FE_time, and BE_time, respectively, for intuitive visualization analysis. Through in-depth analysis of this data, we can more accurately diagnose the health status of the bearings, thus providing a scientific basis for the maintenance of key motion components such as machining centers and robotic arms.

![image-20241129095400634](picture\image-20241129095400634.png)

**Figure 1-3 Bearing Vibration Data [57]**

To enhance the accuracy and robustness of bearing fault detection, this study has augmented the dataset. Specifically, an overlapping sampling method was used to expand the training dataset. This method cleverly designs a certain proportion of overlap between each segment of the signal and its subsequent signal when extracting training samples from the original signal. This overlapping strategy not only retains the continuity and context information of the signal but also effectively increases the size of the dataset, increasing the number of training data by 14.19%. In this way, the model can be exposed to more data variations, thereby improving its generalization ability and fault diagnosis accuracy in practical applications.

![image-20241129100508496](picture\image-20241129100508496.png)

**Figure 1-4 Resampling Data Augmentation Schematic**

## 2 Fault Detection Based on Convolutional Neural Networks

This paper first constructs a CNN-based model for automatic feature extraction and bearing fault classification. The CNN model consists of 5 convolutional pooling layers, with the kernel shapes set to 64, 16, 8, 3, 3 in sequence, aiming to extract increasingly abstract feature representations from the original signal layer by layer. Through these hierarchical convolutional operations, the model can capture local features in the bearing vibration signal and gradually integrate them into a global feature representation, providing strong feature support for fault diagnosis.

Additionally, the total number of training parameters for this model is 54,793, indicating that while the model remains relatively lightweight, it also has enough parameters to learn complex fault patterns. This model design not only improves the efficiency of fault detection but also makes the model easier to train and generalize to different bearing fault datasets.



 ![image-20241129100734345](picture\image-20241129100734345.png)                       

 ![image-20241129100739722](picture\image-20241129100739722.png)

 ![image-20241129100746006](picture\image-20241129100746006.png)

**Figure 2-1 CNN Model Structure**

The proposed CNN model was thoroughly trained to achieve the best bearing fault diagnosis results. The entire training process was set to 20 Epochs, which means the entire dataset was traversed 20 times for learning. At the 8th Epoch, the model's performance reached a significant result, with both the diagnostic accuracy and generalization capability performing well. This observation indicates that the CNN model can quickly converge and capture key features in the bearing fault data after relatively few training cycles. This fast convergence characteristic not only improves the efficiency of model training but also reduces the risk of overfitting, making the model more stable and reliable in practical applications. After the 8th Epoch, we continued to train the model to ensure it maintained good performance at different stages, and ultimately, after all the set Epochs were completed, we obtained a model that performed well in bearing fault diagnosis tasks.



 

![image-20241129100823031](picture\image-20241129100823031.png)![image-20241129100828748](picture\image-20241129100828748.png)

**Figure 2-2 CNN Results, Left is Loss, Right is Accuracy**

## 3 Fault Detection Based on Recurrent Neural Networks

This study also explored two types of recurrent neural network architectures—LSTM [58] and GRU [59], both combined with CNNs, to assess which architecture performs better in bearing fault diagnosis tasks. Below is a detailed description of these two model architectures:

LSTM+CNN Model:

- Blue box (CNN layer): The CNN part of this model consists of multiple convolutional layers, with kernel shapes of 64, 3, 3, 3, 3 respectively. This configuration allows the model to extract local features from the original signal and convert these features into higher-level representations, providing rich feature inputs for subsequent time series analysis.
- Red box (LSTM_layer1): After the CNN layer, we introduce the first LSTM layer, which is used to process time series data and extract temporal features.
- Orange box (LSTM_fc1): This is a fully connected layer located after the LSTM layer, used to further integrate features and prepare for final classification.
- Yellow box (LSTM_layer2): The second LSTM layer, which continues to process the output from the previous LSTM layer to capture more complex temporal dependencies.
- Green box (LSTM_fc2): The last fully connected layer, which converts the output of the LSTM layer into the final classification result.

GRU+CNN Model:

- Blue box (CNN layer): The same as the LSTM+CNN model, the CNN part of the GRU+CNN model also consists of convolutional layers with kernel shapes of 64, 3, 3, 3, 3.
- Red box (GRU_layer1): This is the first layer of the GRU model, which is similar to LSTM but has a simpler structure and fewer parameters, used to process time series data.
- Orange box (GRU_fc1): This is a fully connected layer located after the first GRU layer, used to integrate features.
- Yellow box (GRU_layer2): The second GRU layer, which further processes time series data, capturing deep temporal features.
- Green box (GRU_fc2): The last fully connected layer, which converts the output of the GRU layer into the final classification decision.

 ![image-20241129100857073](picture\image-20241129100857073.png)

**Figure 3-1 LSTM+CNN or GRU+CNN Model**

The CNN part of the above models consists of five convolutional pooling layers, with kernel shapes set to 64, 3, 3, 3, 3 respectively. These layers work together on the input data, effectively extracting spatial features from the bearing vibration signal. This leverages the advantage of CNNs in image processing, extracting spatial features from the input bearing vibration signal through hierarchical feature extraction of convolutional layers. The designed convolutional layers allow the model to capture local patterns and texture information in the signal, providing a rich feature representation for subsequent time series analysis.

After the CNN layer, the model transitions to the LSTM part, which consists of two LSTM layers and two fully connected layers. The first LSTM layer (red box) begins to process time series data, capturing long-term dependencies in the signal. Subsequently, the first fully connected layer (orange box) further integrates the features from the LSTM layer. The second LSTM layer (yellow box) continues to process time series data in depth, and the second fully connected layer (green box) synthesizes this information to provide decision support for the final fault classification. Through the combination of CNN and LSTM, the model can not only capture local features of the bearing vibration signal but also understand the evolution of these features over time, thereby effectively identifying types of bearing faults. During the 20 Epoch training process, the model achieved good diagnostic results by the 4th Epoch, demonstrating the model's rapid convergence and sensitivity to fault characteristics.

![image-20241129100910803](picture\image-20241129100910803.png)  ![image-20241129100915031](picture\image-20241129100915031.png)

 **Figure 3-2 LSTM+CNN Results, Left is Loss, Right is Accuracy**

In the CNN+GRU model, following the CNN part is the GRU network structure. The GRU layer (red box) is responsible for processing time series data and extracting temporal features. GRU is an efficient variant of recurrent neural networks; it controls the flow of information by introducing update gates and reset gates, effectively capturing long-range dependencies. After the GRU layer, a fully connected layer (orange box) is added to further integrate features and provide a richer feature representation for classification. To increase the depth of the model and improve classification accuracy, we added another GRU layer (yellow box) and another fully connected layer (green box) after the first GRU layer. This design enables the model to learn the temporal dynamics of the signal in more detail, further enhancing the accuracy of fault diagnosis. During the 20 Epoch training process, the GRU+CNN model showed good performance by the 3rd Epoch, indicating the model's ability to quickly learn and recognize bearing fault characteristics.



 ![image-20241129100947414](picture\image-20241129100947414.png)![image-20241129100951577](picture\image-20241129100951577.png)

**Figure 3-3 GRU+CNN Results, Left is Loss, Right is Accuracy**

## 4 Fault Detection Based on Self-Attention Mechanism

This paper also employs a novel hybrid model that combines the advantages of Self-Attention with CNNs to improve the performance of bearing fault diagnosis. The number of heads in the self-attention part of the model is set to 8, allowing the model to capture information in parallel in different representation subspaces, enhancing the model's understanding and processing capabilities of data.

![image-20241129101011305](picture\image-20241129101011305.png)

**Figure 4-1 Self-Attention and Multi-Head Self-Attention Structure [61]**

The principle of Self-Attention is to compute representations based on input sequences, associating different positions of a single sequence. This mechanism can capture internal dependencies within the sequence, allowing the model to consider other elements when processing one element, thus better understanding the context of the entire sequence. In Self-Attention, each element calculates an attention score, which determines how much weight each element should be given when generating the output representation [61].

![image-20241129101026006](picture\image-20241129101026006.png)

**Figure 4-2 Self-Attention+CNN Model**

By combining these, our model can not only leverage the powerful spatial feature extraction capabilities of CNNs but also capture deeper sequence features and context information through the self-attention mechanism. This hybrid approach enhances the model's analytical ability for bearing vibration signals, leading to more accurate diagnosis of bearing fault states. Experimental results show that the model has performed excellently in bearing fault diagnosis tasks, proving the effectiveness and potential of combining self-attention with CNNs.

![image-20241129101036788](picture\image-20241129101036788.png)![image-20241129101041393](picture\image-20241129101041393.png)

**Figure 4-3 Self-Attention+CNN Results, Left is Loss, Right is Accuracy**

The model achieved good diagnostic results by the 3rd Epoch during the training process, demonstrating the model's rapid learning ability and efficiency.

By combining these, the model can not only utilize the powerful spatial feature extraction capabilities of CNNs but also capture deeper sequence features and context information through the self-attention mechanism. This hybrid approach enhances the model's analytical ability for bearing vibration signals, leading to more accurate diagnosis of bearing fault states. Experimental results show that the model has performed excellently in bearing fault diagnosis tasks, proving the effectiveness and potential of combining self-attention with CNNs.

## 5 Model Evaluation

**Table 4-1 Processing Time of 100 Workpieces on 4 Machines**

| **Model**             | **CNN**   | **LSTM+CNN** | **GRU+CNN** | **Self-Attention+CNN** |
| --------------------- | --------- | ------------ | ----------- | ---------------------- |
| **Training Time/(S)** | **21.00** | **855.73**   | **729.28**  | **31.58**              |
| **Accuracy**          | **High**  | **High**     | **High**    | **High**               |

A comprehensive comparison of the four different models mentioned above was conducted, including traditional CNN models, LSTM+CNN models, GRU+CNN models, and Self-Attention+CNN models. The comparison indicators include training time, accuracy, and the number of Epochs to convergence. Through this comparative analysis, it can be seen that the Self-Attention+CNN model performs excellently in multiple aspects.

1. Training Time: Due to its efficient parallel computing capability, the Self-Attention+CNN model shows a shorter training time compared to sequential models such as LSTM and GRU. This gives it an advantage in practical applications, especially in scenarios that require rapid deployment and response.
2. Accuracy: In terms of accuracy, the Self-Attention+CNN model also performs well. By using the self-attention mechanism, the model can better capture long-range dependencies in bearing vibration signals, which is particularly important for fault patterns with complex temporal characteristics. Additionally, the introduction of the CNN part further enhances the model's ability to extract spatial features, significantly improving the overall fault diagnosis accuracy.
3. Convergence Epochs: In terms of model convergence speed, the Self-Attention+CNN model achieved good results by the 3rd Epoch, showing rapid learning capabilities. This characteristic means the model can adapt to new data in a short time, which is a very valuable feature for online learning and real-time monitoring systems.

Considering the above factors, the Self-Attention+CNN model shows its unique advantages in bearing fault diagnosis tasks. It can not only quickly learn and accurately identify different fault types but also complete training in a short time, which is significant for improving production efficiency and reducing maintenance costs. Therefore, this paper believes that the Self-Attention+CNN model is a powerful tool worth further research and application.





# 基于CNN、LSTM、GRU、Self-Attention的轴承故障检测

### 1 设备数据采集与增强

在加工中心和机械臂等运动部件的旋转部分，轴承扮演着至关重要的角色。轴承的健康状况直接影响到加工中心和机械臂的运行效率和精度，因此对其进行故障检测尤为重要。轴承的常见故障类型包括内圈故障、外圈故障和滚珠故障。这些故障的发生，通常是由于滚珠与外圈或内圈的局部损坏部分相互撞击，或者损坏的滚珠与内外圈发生撞击，从而激发出轴承与响应传感器之间的高频共振。通过使用加速度计采集轴承的振动信号，利用机器学习、深度学习的方法对采集的数据进行识别判断，可以有效地实现对这些故障的诊断。

​              ![image-20241129101151134](picture\image-20241129101151134.png)                 

图1-1 轴承结构示意图

此外，对机械臂的运行温度、电机驱动电压和电流以及位置位姿等参数的监测，也是判断其运行状态的重要手段。通过收集这些数据，并分析其所处的范围及发展趋势，可以及时识别潜在的问题，从而采取相应的维护措施，确保加工中心的稳定运行。

本文采用了普渡大学开源的轴承故障诊断数据集[57]，该数据集包含了九种不同故障类型的振动信号数据。每种故障类型均涵盖了驱动端（DE_time）、风扇端（FE_time）和基座（BE_time）三个维度的测量数据，为研究提供了全面的数据分析基础。

​                               ![image-20241129095324236](picture\image-20241129095324236.png)

图1-2 普渡大学轴承故障检测试验台[57]

具体来说，数据集中的每种故障类型都包含了约85,000条数据，如图4-4，这些数据以红色、绿色和蓝色分别表示DE_time、FE_time和BE_time，以便进行直观的可视化分析。通过对这些数据的深入分析，我们可以更准确地诊断轴承的健康状况，从而为加工中心机械臂等关键运动部件的维护提供科学依据。

​     ![image-20241129095400634](picture\image-20241129095400634.png)

​     

图1-3 轴承振动数据[57]

为了提升轴承故障检测的准确性和鲁棒性，本研究对数据集进行了增强处理。具体而言，采用了重叠采样的方法来扩充训练数据集。该方法在从原始信号中提取训练样本时，巧妙地设计了每一段信号与其后继信号之间存在一定比例的重叠部分。这种重叠策略不仅保留了信号的连续性和上下文信息，而且有效地增加了数据集的大小，使训练数据的数量提升了14.19%。通过这种方式，模型能够接触到更多的数据变化，从而提高了其在实际应用中的泛化能力和故障诊断的准确性。

 ![image-20241129100508496](picture\image-20241129100508496.png)

图1-4 重采样数据增强示意图

### 2 基于卷积神经网络的故障检测

本文首先构建了一个基于CNN的模型，用于自动特征提取和轴承故障分类。该CNN模型包含5个卷积池化层，每层的核形状依次设置为64、16、8、3、3，这样的配置旨在从原始信号中逐层提取出越来越抽象的特征表示。通过这些层次化的卷积操作，模型能够捕捉到轴承振动信号中的局部特征，并逐渐整合成全局性的特征表征，为故障诊断提供了强有力的特征支持。

此外，该模型的总训练参数数量为54,793，这表明模型在保持相对轻量化的同时，也具备了足够的参数来学习复杂的故障模式。这一模型设计不仅提高了故障检测的效率，也使得模型更易于训练和泛化到不同的轴承故障数据集上。

​        ![image-20241129100734345](picture\image-20241129100734345.png)                       

 ![image-20241129100739722](picture\image-20241129100739722.png)

 ![image-20241129100746006](picture\image-20241129100746006.png)

图2-1 CNN模型结构

对所提出的CNN模型进行了充分的训练，以达到最佳的轴承故障诊断效果。整个训练过程共设置了20个Epoch，即对整个数据集进行了20轮的遍历学习。在第8个Epoch时，模型的性能达到了一个显著的结果，此时模型的诊断准确率和泛化能力均表现良好。这一观察结果表明，该CNN模型在经过相对较少的训练周期后，就能快速收敛并捕捉到轴承故障数据中的关键特征。这种快速收敛的特性不仅提高了模型训练的效率，也减少了过拟合的风险，使得模型在实际应用中更加稳定和可靠。在第8个Epoch之后，我们继续训练模型，以确保其在不同阶段都能保持较好的性能，最终在所有设定的Epoch完成后，我们获得了一个在轴承故障诊断任务上表现较好的模型。

   ![image-20241129100823031](picture\image-20241129100823031.png)![image-20241129100828748](picture\image-20241129100828748.png)

图2-2 CNN结果示意，左为Loss，右为Accuracy

### 3 基于循环神经网络的故障检测

本研究还探索了两种循环神经网络架构——LSTM [58]和GRU [59]，它们均与CNN结合，以评估哪种架构在轴承故障诊断任务上表现更优。以下是对这两种模型架构的详细描述：

LSTM+CNN模型：

- 蓝色方框（CNN层）： 该模型的CNN部分由多个卷积层组成，各层的核形状分别为64、3、3、3、3。这种配置使得模型能够从原始信号中提取局部特征，并将这些特征转换为更高级的表示，为后续的时序分析提供丰富的特征输入。
- 红色方框（LSTM_layer1）： 在CNN层之后，我们引入了第一个LSTM层，用于处理时间序列数据并提取时序特征。
- 橙色方框（LSTM_fc1）： 这是一个全连接层，位于LSTM层之后，用于进一步整合特征并为最终分类做准备。
- 黄色方框（LSTM_layer2）： 第二个LSTM层，它继续处理来自前一个LSTM层的输出，以捕捉更复杂的时序依赖关系。
- 绿色方框（LSTM_fc2）： 最后一个全连接层，它将LSTM层的输出转换为最终的分类结果。

GRU+CNN模型：

- 蓝色方框（CNN层）： 与LSTM+CNN模型相同，GRU+CNN模型的CNN部分也由核形状为64、3、3、3、3的卷积层组成。
- 红色方框（GRU_layer1）： 这是GRU模型的第一个层，它与LSTM类似，但结构更简单，参数更少，用于处理时间序列数据。
- 橙色方框（GRU_fc1）： 这是一个全连接层，位于第一个GRU层之后，用于整合特征。
- 黄色方框（GRU_layer2）： 第二个GRU层，它进一步处理时间序列数据，捕捉深层次的时序特征。
- 绿色方框（GRU_fc2）： 最后一个全连接层，它将GRU层的输出转换为最终的分类决策。

 ![image-20241129100857073](picture\image-20241129100857073.png)

图3-1 LSTM+CNN或GRU+CNN模型示意

上述模型的CNN部分均由五个卷积池化层组成，每层的卷积核形状分别设置为64、3、3、3、3，这些层共同作用于输入数据，有效地提取了轴承振动信号中的空间特征。这利用了CNN在图像处理中的优势，通过卷积层的层次化特征提取，从输入的轴承振动信号中提取空间特征。通过设计的卷积层，模型能够捕捉到信号中的局部模式和纹理信息，为后续的时序分析提供了丰富的特征表示。

在CNN层之后，模型转入LSTM部分，这一部分由两个LSTM层和两个全连接层组成。第一个LSTM层（红色方框）开始处理时间序列数据，捕捉信号中的长期依赖关系。随后，第一个全连接层（橙色方框）对LSTM层的输出进行进一步的特征整合。第二个LSTM层（黄色方框）继续深入处理时间序列数据，而第二个全连接层（绿色方框）则将这些信息综合起来，为最终的故障分类提供决策支持。通过这种CNN和LSTM的结合，模型不仅能够捕捉到轴承振动信号的局部特征，还能够理解这些特征随时间的演变，从而实现对轴承故障类型的有效识别。在20个Epoch的训练过程中，模型在第4个Epoch就达到了较好的诊断效果，这表明了模型的快速收敛性和对故障特征的敏感性。

 ![image-20241129100910803](picture\image-20241129100910803.png)  ![image-20241129100915031](picture\image-20241129100915031.png)

图3-2 LSTM+CNN结果，左为Loss，右为Accuracy

在CNN+GRU模型紧接着CNN部分的是GRU网络结构。GRU层（红色方框）的作用是处理时间序列数据并提取时序特征。GRU是一种高效的循环神经网络变体，它通过引入更新门和重置门来控制信息的流动，从而有效地捕捉长距离依赖关系。在GRU层之后，加入了一个全连接层（橙色方框），进一步整合特征，为分类提供更丰富的特征表示。为了增加模型的深度并提高分类的准确性，我们在第一个GRU层之后又添加了一个GRU层（黄色方框），以及另一个全连接层（绿色方框）。这样的设计使得模型能够更细致地学习信号中的时间动态变化，进一步提升了故障诊断的准确性。在20个Epoch的训练过程中，GRU+CNN模型在第3个Epoch就展现出了较好的性能，这表明模型能够快速学习并识别轴承的故障特征。

   ![image-20241129100947414](picture\image-20241129100947414.png)![image-20241129100951577](picture\image-20241129100951577.png)

图3-3 GRU+CNN结果示意，左为Loss，右为Accuracy

### 4 基于自注意力机制的故障检测

本文还采用了一种新颖的混合模型，该模型将Self-Attention与CNN的优势相结合[60]，以提高轴承故障诊断的性能。模型中自注意力部分的多头数设置为8，这允许模型在不同表示子空间中并行地捕捉信息，增强了模型对数据的理解和处理能力。

 ![image-20241129101011305](picture\image-20241129101011305.png)

图4-1 self-attention和multi heads self-attention结构示意[61]

Self-Attention的原理是基于输入序列计算表示，它将单个序列的不同位置关联起来。这种机制能够捕捉序列内部的依赖关系，允许模型在处理一个元素时考虑到其他元素，从而更好地理解整个序列的上下文信息。在Self-Attention中，每个元素都会计算出一个注意力分数，这些分数决定了在生成输出表示时，每个元素应该被赋予多少权重[61]。

 ![image-20241129101026006](picture\image-20241129101026006.png)

图4-2 Self-Attention+CNN模型示意

通过这种结合，我们的模型不仅能够利用CNN强大的空间特征提取能力，还能够通过自注意力机制捕捉到更深层次的序列特征和上下文信息。这种混合方法提高了模型对轴承振动信号的分析能力，从而更准确地诊断轴承的故障状态。实验结果表明，该模型在轴承故障诊断任务上表现出了优异的性能，证明了自注意力与CNN结合的有效性和潜力。

   ![image-20241129101036788](picture\image-20241129101036788.png)![image-20241129101041393](picture\image-20241129101041393.png)

图4-3 Self-Attention+CNN结果示意，左为Loss，右为Accuracy

该模型在训练过程中，在第3个Epoch时，便达到了较好的诊断效果，显示出模型的快速学习能力和高效性。

通过这种结合，该模型不仅能够利用CNN强大的空间特征提取能力，还能够通过自注意力机制捕捉到更深层次的序列特征和上下文信息。这种混合方法提高了模型对轴承振动信号的分析能力，从而更准确地诊断轴承的故障状态。实验结果表明，该模型在轴承故障诊断任务上表现出了优异的性能，证明了Self-Attention与CNN结合的有效性和潜力。

### 5 模型评估

表4-1 100个工件在4个机器上的加工时间

| **模型**         | **CNN**    | **LSTM+CNN** | **GRU+CNN** | **Self-Attention+CNN** |
| ---------------- | ---------- | ------------ | ----------- | ---------------------- |
| **训练时间/(S)** | **21. 00** | **855.73**   | **729.28**  | **31.58**              |
| **准确率**       | **较高**   | **高**       | **高**      | **高**                 |

对上述四种不同的模型进行了综合比较，包括传统的CNN模型、LSTM+CNN模型、GRU+CNN模型以及Self-Attention+CNN模型。比较的指标包括训练时间、准确率和收敛的Epoch数。通过这一对比分析，可以发现Self-Attention+CNN模型在多个方面表现出色。

1. 训练时间：Self-Attention+CNN模型由于其高效的并行计算能力，相较于序列模型如LSTM和GRU，展现出了更短的训练时间。这使得它在实际应用中更具优势，尤其是在需要快速部署和响应的场景中。
2. 准确率：在准确率方面，Self-Attention +CNN模型同样表现不俗。通过自注意力机制，模型能够更好地捕捉轴承振动信号中的长距离依赖关系，这对于那些具有复杂时序特性的故障模式尤为重要。此外，CNN部分的引入进一步增强了模型对空间特征的提取能力，使得整体的故障诊断准确率得到了显著提升。
3. 收敛Epoch：在模型的收敛速度上，Self-Attention +CNN模型在第3个Epoch时便达到了较好的效果，显示出了快速的学习能力。这一特性意味着模型能够在短时间内适应新的数据，对于在线学习和实时监测系统来说，这是一个非常有价值的特性。

综合考虑以上因素，Self-Attention +CNN模型在轴承故障诊断任务中展现了其独特的优势。它不仅能够快速学习并准确识别出不同的故障类型，还能够在较短的时间内完成训练，这对于提高生产效率和降低维护成本具有重要意义。因此，本文认为Self-Attention +CNN模型是一个值得进一步研究和应用的有力工具。
