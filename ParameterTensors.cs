namespace llm.cs;

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the parameters of the model
public class ParameterTensors
{
    public (int,int) Wte; // (V, C)
    public (int,int) Wpe; // (maxT, C)
    public (int,int) Ln1w; // (L, C)
    public (int,int) Ln1b; // (L, C)
    public (int,int) Qkvw; // (L, 3*C, C)
    public (int,int) Qkvb; // (L, 3*C)
    public (int,int) AttProjw; // (L, C, C)
    public (int,int) AttProjb; // (L, C)
    public (int,int) Ln2w; // (L, C)
    public (int,int) Ln2b; // (L, C)
    public (int,int) Fcw; // (L, 4*C, C)
    public (int,int) Fcb; // (L, 4*C)
    public (int,int) FcProjw; // (L, C, 4*C)
    public (int,int) FcProjb; // (L, C)
    public (int,int) Lnfw; // (C)
    public (int,int) Lnfb; // (C)
}
