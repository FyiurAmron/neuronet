package vax.snnt.neuronet;

/**

 @author toor
 */
public class Neuron {
    private final TransferFunction inputTransferFunction, outputTransferFunction;

    private double potential = 0;

    public Neuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction ) {
        this.inputTransferFunction = inputTransferFunction;
        this.outputTransferFunction = outputTransferFunction;
    }

    public Neuron ( Neuron neuron ) {
        this( neuron.inputTransferFunction.copy(), neuron.outputTransferFunction.copy() );
    }

    public void add ( double inputPotential ) {
        potential += inputTransferFunction.f( inputPotential );
    }

    /*
     public void set ( double inputPotential ) {
     potential = inputTransferFunction.f( inputPotential );
     }
     */
    public double getPotential () {
        return potential;
    }

    public void preProcess () {
        potential = outputTransferFunction.f( potential );
    }

    public void postProcess () {
        potential = 0;
    }

    public Neuron copy () {
        return new Neuron( this );
    }
}
