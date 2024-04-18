namespace llm.cs;

public class GPT2
{
    public const int NUM_PARAMETER_TENSORS = 16;
    public const int NUM_ACTIVATION_TENSORS = 23;

    public GPT2Config Config = new();
    // the weights of the model, and their sizes
    public ParameterTensors Params = new();
    public int[] ParamSizes = new int[NUM_PARAMETER_TENSORS];
    public float[] ParamMemory;
    public int NumParameters;
    // gradients of the weights
    public ParameterTensors Grads = new();
    public float[] GradsMemory;
    // buffers for the AdamW optimizer
    public float[] M_Memory;
    public float[] V_Memory;
    // the activations of the model, and their sizes
    public ActivationTensors Acts = new();
    public int[] ActSizes = new int[NUM_ACTIVATION_TENSORS];
    public float[] ActsMemory;
    public int NumActivations;
    // gradients of the activations
    public ActivationTensors GradsActs = new();
    public float[] GradsActsMemory;
    // other run state configuration
    public int BatchSize; // the batch size (B) of current forward pass
    public int SeqLen; // the sequence length (T) of current forward pass
    public int[] Inputs; // the input tokens for the current forward pass
    public int[] Targets; // the target tokens for the current forward pass
    public float MeanLoss; // after a forward pass with targets, will be populated with the mean loss
}

