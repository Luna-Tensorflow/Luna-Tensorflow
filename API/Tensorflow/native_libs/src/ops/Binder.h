//
// Created by wojtek on 14.02.19.
//

#ifndef TFL_BINDER_H
#define TFL_BINDER_H

#include "../graph/GraphSession.h"

class GraphSession;

class Hashable {
public:
	virtual size_t hashcode() const = 0;
};

// TODO rename to Node or something like that
/*
 *  Each Node has a set of Outputs associated with it,
 *  that correspond to outputs of the Tensorflow Node this object represents.
 *  Each Output has a reference to their Node, so that the Node exists in memory
 *  as long as at least one of its Outputs may still be used somewhere
 *  (because the Node is what computes the Output's value in TF Graph).
 *
 *  The Node should store a weak_ptr to its Outputs
 *  so that it can know whether they are still allocated or not,
 *  but Node's lifetime should be bound to the Outputs
 *  in such a way that it won't prevent from their deallocation.
 */
class Binder : public Hashable {
protected:
    Binder() = default;

public:
    /*
     * This method should add the Node to the provided graph
     * AND add all of the Node's Outputs that haven't been freed yet as well.
     */
    virtual void add_to_graph(GraphSession &graph) = 0;

    virtual ~Binder() = default;
};

#endif //TFL_BINDER_H
