package vax.snnt.neuronet;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.function.DoubleConsumer;

/**

 @author toor
 */
public abstract class Neuron implements Processable {

    static public class MidNeuron extends Neuron {
        final private List<NeuronOutput> neuronOutputs;

        public MidNeuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction ) {
            this( inputTransferFunction, outputTransferFunction, new LinkedList<>() );
        }

        private /* public */ MidNeuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction,
                        List<NeuronOutput> neuronOutputs ) {
            super( inputTransferFunction, outputTransferFunction );
            this.neuronOutputs = neuronOutputs;
        }

        @Override
        public void releaseImpl () {
            for( NeuronOutput neuronOutput : neuronOutputs ) {
                neuronOutput.release();
            }
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

    }

    static public class OutputNeuron extends Neuron {
        private final DoubleConsumer outputConsumer;

        public OutputNeuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction,
                DoubleConsumer outputConsumer ) {
            super( inputTransferFunction, outputTransferFunction );
            this.outputConsumer = outputConsumer;
        }

        @Override
        public void releaseImpl () {
            outputConsumer.accept( getPotential() );
        }

    }

    // final private List<NeuronOutput> neuronInputs;
    final private TransferFunction inputTransferFunction, outputTransferFunction;
    private double potential = 0;

    public Neuron ( TransferFunction inputTransferFunction, TransferFunction outputTransferFunction ) {
        this.inputTransferFunction = inputTransferFunction;
        this.outputTransferFunction = outputTransferFunction;
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

    public abstract void releaseImpl ();

    @Override
    public void process () {
        potential = outputTransferFunction.f( potential );
        releaseImpl();
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
