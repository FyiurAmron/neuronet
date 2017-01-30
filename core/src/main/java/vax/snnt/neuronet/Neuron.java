package vax.snnt.neuronet;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.function.DoubleConsumer;

/**

 @author toor
 */
public class Neuron implements Processable {
    // final private List<NeuronOutput> neuronInputs;
    private final List<NeuronOutput> neuronOutputs;
    private final TransferFunction inputTransferFunction, outputTransferFunction;
    transient private final DoubleConsumer outputConsumer;

    private double potential = 0;

    public Neuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction ) {
        this( inputTransferFunction, outputTransferFunction, null );
    }

    public Neuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction, DoubleConsumer outputConsumer ) {
        this( inputTransferFunction, outputTransferFunction, outputConsumer, new LinkedList<>() );
    }

    public Neuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction,
            DoubleConsumer outputConsumer, List<NeuronOutput> neuronOutputs ) {
        this.neuronOutputs = neuronOutputs;
        this.outputConsumer = outputConsumer;
        this.inputTransferFunction = inputTransferFunction;
        this.outputTransferFunction = outputTransferFunction;
    }

    public Neuron mutate ( double ampChange, double shiftChange, double weightChange ) {
        TransferFunction itf = inputTransferFunction.copy();
        itf.ampIn += ampChange;
        itf.ampOut += ampChange;
        itf.shiftIn += shiftChange;
        itf.shiftOut += shiftChange;

        TransferFunction otf = outputTransferFunction.copy();
        otf.ampIn += ampChange;
        otf.ampOut += ampChange;
        otf.shiftIn += shiftChange;
        otf.shiftOut += shiftChange;

        // TODO weights
        Neuron neu = new Neuron( inputTransferFunction, outputTransferFunction );
        for( NeuronOutput neuronOutput : neuronOutputs ) {
            neu.addNeuronOutput( neuronOutput.neuron, neuronOutput.weight );
        }
        return neu;
    }

    public void addNeuronOutput ( Neuron n, double weight ) {
        neuronOutputs.add( new NeuronOutput( n, weight ) );
    }

    public void addNeuronOutputs ( List<Neuron> ns, double[] weights ) {
        if ( ns.isEmpty() || ns.size() != weights.length ) {
            throw new IllegalArgumentException();
        }

        Iterator<Neuron> it = ns.listIterator();
        int i = 0;
        for( Neuron n = it.next(); it.hasNext(); n = it.next() ) {
            neuronOutputs.add( new NeuronOutput( n, weights[i] ) );
            i++;
        }
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

    @Override
    public void process () {
        potential = outputTransferFunction.f( potential );
        for( NeuronOutput neuronOutput : neuronOutputs ) {
            neuronOutput.release();
        }
        if ( outputConsumer != null ) {
            outputConsumer.accept( potential );
        }
        potential = 0;
    }

    public class NeuronOutput {
        private final Neuron neuron;
        private final double weight;

        public NeuronOutput ( Neuron neuron, double weight ) {
            this.neuron = neuron;
            this.weight = weight;
        }

        public void release () {
            neuron.add( potential * weight );
        }
    }
}
