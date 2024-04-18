namespace llm.cs;

public class GPT2Config
{
    public int MaxSeqLen; // max sequence length, e.g. 1024
    public int VocabSize; // vocab size, e.g. 50257
    public int NumLayers; // number of layers, e.g. 12
    public int NumHeads; // number of heads in attention, e.g. 12
    public int Channels; // number of channels, e.g. 768
}
