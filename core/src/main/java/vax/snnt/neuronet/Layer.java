package vax.snnt.neuronet;

import java.util.ArrayList;
import java.util.List;

/**

 @author toor
 */
public class Layer /* implements Processable */ {
    private final ArrayList<Integer> neurons = new ArrayList<>(); // TODO switch to indexes from Network array

    /*
     public Layer ( List<Neuron> neurons ) {
     this.neurons = neurons;
     }
     */
    public Layer ( Layer layer ) {
        neurons.addAll( layer.neurons );
    }

    public Layer () {
    }

    public void addNeuron ( int neuronNr ) {
        neurons.add( neuronNr );
    }

    /*
     @Override
     public void process () {
     for( Neuron neuron : neurons ) {
     neuron.process();
     }
     }
     */
    public List<Integer> getNeurons () {
        return neurons;
    }

    public Layer copy () {
        return new Layer( this );
    }
}
